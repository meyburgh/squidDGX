#pragma once

// Grammar-constrained JSON decoding for OpenAI `response_format`.
//
// JsonSchema compiles a JSON Schema (or the implicit "any JSON object" of json_object) into a
// node table. JsonMatcher is a byte-level pushdown matcher over that table: it answers "can the
// output so far be extended by these bytes and still be a prefix of a valid document?" and
// "is the document complete?". It is a fixed-size value type so a candidate token can be
// tested on a copy.
//
// Supported keywords: type (string or array), properties, required, additionalProperties
// (bool or schema), items, enum, const (scalars), anyOf, oneOf (treated as anyOf),
// allOf with one entry, $ref to "#", "#/$defs/<name>" or "#/definitions/<name>".
// Ignored (not enforced): string formats/patterns/lengths, numeric ranges, minItems/maxItems.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace squid {

enum : uint8_t {
    kJsObj = 1, kJsArr = 2, kJsStr = 4, kJsInt = 8, kJsNum = 16, kJsBool = 32, kJsNull = 64,
    kJsAllTypes = 127,
};

constexpr int32_t kJsForbidden = -1;
constexpr int32_t kJsAnyNode = 0;  // node 0 always accepts any JSON value

struct JsonSchemaNode {
    uint8_t types = kJsAllTypes;
    std::vector<std::string> prop_keys;   // canonical JSON-encoded names, with quotes
    std::vector<uint64_t> prop_hash;      // hash of the encoded bytes between the quotes
    std::vector<int32_t> prop_schema;
    uint64_t required = 0;                // bit i => prop i required (first 64 props)
    int32_t additional = kJsAnyNode;      // kJsForbidden, kJsAnyNode or a node index
    int32_t items = kJsAnyNode;
    std::vector<int32_t> alts;            // anyOf / oneOf members; non-empty => union node
    std::vector<std::string> literals;    // enum / const, serialized; non-empty => must match one
    int32_t ref = -1;                     // alias to another node
    std::string ref_path;                 // unresolved $ref
};

struct JsonSchema {
    std::vector<JsonSchemaNode> nodes;
    int32_t root = kJsAnyNode;

    // json_object: the document must be an object with any contents.
    void init_any_object();
    // json_schema: returns false (with err) if the schema text is not valid JSON.
    bool compile(std::string_view schema_json, std::string* err);

    // Tool calls: start empty and add one entry per tool; each entry matches
    // {"name": <that name>, "args": <its parameters>}. finish_tools() sets the root to a fallback
    // used when the caller cannot tell which tool is being called: the name must be one of the
    // tools and args any object (a union over many tools would branch the matcher per tool).
    void init_empty();
    int32_t add_tool(std::string_view name, std::string_view parameters_json);  // returns the entry node
    void finish_tools(const std::vector<std::string>& names);

    int32_t resolve(int32_t idx) const;
};

uint64_t json_key_hash(std::string_view encoded_inner);

class JsonMatcher {
public:
    static constexpr int kMaxDepth = 32;
    static constexpr int kMaxStacks = 8;

    // Plain-old-data on purpose: matchers are copied per candidate token, and default member
    // initializers would zero 8 x 32 frames on every copy.
    struct Frame {
        uint8_t kind;
        uint8_t st;
        uint8_t aux;
        uint8_t pad;
        int32_t node;
        uint32_t a;   // pos / count
        int32_t b;    // current prop index (objects) / literal id (true/false/null)
        uint64_t seen;
        uint64_t h;
        uint64_t set[4];
    };
    struct Stack {
        uint16_t depth;
        uint16_t ws;
        bool done;   // root value finished
        Frame f[kMaxDepth];
    };

    JsonMatcher() = default;
    JsonMatcher(const JsonMatcher& o) { *this = o; }
    JsonMatcher& operator=(const JsonMatcher& o);  // copies only the live stacks and frames

    void init(const JsonSchema* schema);
    void init(const JsonSchema* schema, int32_t root);
    bool advance(std::string_view bytes);              // false => rejected (state unchanged)
    bool accepts(std::string_view bytes) const;
    bool can_end() const;                              // the bytes so far form a complete document
    bool started() const { return started_; }

private:
    const JsonSchema* schema_ = nullptr;
    uint8_t n_ = 0;
    bool started_ = false;
    Stack stacks_[kMaxStacks];   // only [0, n_) are meaningful

    bool step_byte(uint8_t c);
    void feed(const Stack& s, uint8_t c, Stack* out, int& n_out, int depth_guard) const;
    static void copy_stack(Stack& dst, const Stack& src);
    bool stack_can_end(const Stack& s) const;
};

} // namespace squid
