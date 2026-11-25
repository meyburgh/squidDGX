#include "json_constraint.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace squid {

namespace {

// ---------------------------------------------------------------------------------------------
// Minimal JSON DOM, used only to read the schema.
// ---------------------------------------------------------------------------------------------
struct JVal {
    enum Type : uint8_t { Null, Bool, Num, Str, Arr, Obj } t = Null;
    std::string s;  // decoded string, raw number text, or "true"/"false"
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    const JVal* get(std::string_view k) const {
        if (t != Obj) return nullptr;
        for (const auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct JParser {
    std::string_view in;
    size_t p = 0;
    bool ok = true;

    void ws() { while (p < in.size() && (in[p] == ' ' || in[p] == '\t' || in[p] == '\n' || in[p] == '\r')) ++p; }

    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o.push_back(static_cast<char>(cp));
        else if (cp < 0x800) { o.push_back(static_cast<char>(0xC0 | (cp >> 6))); o.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { o.push_back(static_cast<char>(0xE0 | (cp >> 12))); o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); o.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else { o.push_back(static_cast<char>(0xF0 | (cp >> 18))); o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F))); o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); o.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    }
    bool hex4(uint32_t* out) {
        if (p + 4 > in.size()) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = in[p++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return false;
        }
        *out = v;
        return true;
    }
    bool str(std::string& o) {
        if (p >= in.size() || in[p] != '"') return false;
        ++p;
        while (p < in.size()) {
            const char c = in[p++];
            if (c == '"') return true;
            if (c != '\\') { o.push_back(c); continue; }
            if (p >= in.size()) return false;
            const char e = in[p++];
            switch (e) {
                case '"': o.push_back('"'); break;
                case '\\': o.push_back('\\'); break;
                case '/': o.push_back('/'); break;
                case 'b': o.push_back('\b'); break;
                case 'f': o.push_back('\f'); break;
                case 'n': o.push_back('\n'); break;
                case 'r': o.push_back('\r'); break;
                case 't': o.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(&cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 6 <= in.size() && in[p] == '\\' && in[p + 1] == 'u') {
                        p += 2;
                        uint32_t lo = 0;
                        if (!hex4(&lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(o, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool val(JVal& v, int depth) {
        if (depth > 64) return false;
        ws();
        if (p >= in.size()) return false;
        const char c = in[p];
        if (c == '{') {
            v.t = JVal::Obj; ++p; ws();
            if (p < in.size() && in[p] == '}') { ++p; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (p >= in.size() || in[p] != ':') return false;
                ++p;
                JVal child;
                if (!val(child, depth + 1)) return false;
                v.obj.emplace_back(std::move(k), std::move(child));
                ws();
                if (p < in.size() && in[p] == ',') { ++p; continue; }
                if (p < in.size() && in[p] == '}') { ++p; return true; }
                return false;
            }
        }
        if (c == '[') {
            v.t = JVal::Arr; ++p; ws();
            if (p < in.size() && in[p] == ']') { ++p; return true; }
            for (;;) {
                JVal child;
                if (!val(child, depth + 1)) return false;
                v.arr.push_back(std::move(child));
                ws();
                if (p < in.size() && in[p] == ',') { ++p; continue; }
                if (p < in.size() && in[p] == ']') { ++p; return true; }
                return false;
            }
        }
        if (c == '"') { v.t = JVal::Str; return str(v.s); }
        if (in.substr(p, 4) == "true") { v.t = JVal::Bool; v.s = "true"; p += 4; return true; }
        if (in.substr(p, 5) == "false") { v.t = JVal::Bool; v.s = "false"; p += 5; return true; }
        if (in.substr(p, 4) == "null") { v.t = JVal::Null; p += 4; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const size_t st = p;
            while (p < in.size() && (in[p] == '-' || in[p] == '+' || in[p] == '.' || in[p] == 'e' || in[p] == 'E' || (in[p] >= '0' && in[p] <= '9'))) ++p;
            v.t = JVal::Num; v.s = std::string(in.substr(st, p - st));
            return true;
        }
        return false;
    }
};

std::string encode_json_string(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 2);
    o.push_back('"');
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    o += "\\u00";
                    o.push_back(kHex[c >> 4]);
                    o.push_back(kHex[c & 15]);
                } else {
                    o.push_back(ch);
                }
        }
    }
    o.push_back('"');
    return o;
}

bool serialize_scalar(const JVal& v, std::string* out) {
    switch (v.t) {
        case JVal::Str: *out = encode_json_string(v.s); return true;
        case JVal::Num: *out = v.s; return true;
        case JVal::Bool: *out = v.s; return true;
        case JVal::Null: *out = "null"; return true;
        default: return false;
    }
}

uint8_t type_bits(std::string_view t) {
    if (t == "object") return kJsObj;
    if (t == "array") return kJsArr;
    if (t == "string") return kJsStr;
    if (t == "integer") return kJsInt;
    if (t == "number") return kJsNum;
    if (t == "boolean") return kJsBool;
    if (t == "null") return kJsNull;
    return 0;
}

struct Compiler {
    JsonSchema& sch;
    std::unordered_map<std::string, int32_t> defs;

    int32_t new_node() {
        sch.nodes.emplace_back();
        return static_cast<int32_t>(sch.nodes.size() - 1);
    }

    void collect_defs(const JVal& v) {
        for (const char* key : {"$defs", "definitions"}) {
            const JVal* d = v.get(key);
            if (!d || d->t != JVal::Obj) continue;
            for (const auto& kv : d->obj) {
                // Reserve the slot first so recursive refs inside the def resolve to it.
                const int32_t idx = new_node();
                defs[std::string(key) + "/" + kv.first] = idx;
            }
        }
        for (const char* key : {"$defs", "definitions"}) {
            const JVal* d = v.get(key);
            if (!d || d->t != JVal::Obj) continue;
            for (const auto& kv : d->obj) {
                const int32_t target = defs[std::string(key) + "/" + kv.first];
                const int32_t body = compile(kv.second, 0);
                sch.nodes[static_cast<size_t>(target)].ref = body;
            }
        }
    }

    int32_t compile(const JVal& v, int depth) {
        if (depth > 64) return kJsAnyNode;
        if (v.t == JVal::Bool) {
            if (v.s == "true") return kJsAnyNode;
            const int32_t idx = new_node();
            sch.nodes[static_cast<size_t>(idx)].types = 0;
            return idx;
        }
        if (v.t != JVal::Obj) return kJsAnyNode;

        const int32_t idx = new_node();
        if (const JVal* r = v.get("$ref"); r && r->t == JVal::Str) {
            sch.nodes[static_cast<size_t>(idx)].ref_path = r->s;
            return idx;
        }

        for (const char* key : {"anyOf", "oneOf"}) {
            const JVal* a = v.get(key);
            if (a && a->t == JVal::Arr && !a->arr.empty()) {
                std::vector<int32_t> alts;
                for (const auto& m : a->arr) alts.push_back(compile(m, depth + 1));
                sch.nodes[static_cast<size_t>(idx)].alts = std::move(alts);
                return idx;
            }
        }
        if (const JVal* a = v.get("allOf"); a && a->t == JVal::Arr && a->arr.size() == 1) {
            const int32_t inner = compile(a->arr[0], depth + 1);
            sch.nodes[static_cast<size_t>(idx)].ref = inner;
            return idx;
        }

        uint8_t types = 0;
        if (const JVal* t = v.get("type")) {
            if (t->t == JVal::Str) types = type_bits(t->s);
            else if (t->t == JVal::Arr) for (const auto& e : t->arr) if (e.t == JVal::Str) types |= type_bits(e.s);
        }
        const JVal* props = v.get("properties");
        const JVal* items = v.get("items");
        if (types == 0) {
            if (props) types = kJsObj;
            else if (items) types = kJsArr;
            else types = kJsAllTypes;
        }
        if (const JVal* n = v.get("nullable"); n && n->t == JVal::Bool && n->s == "true") types |= kJsNull;

        std::vector<std::string> keys;
        std::vector<uint64_t> hashes;
        std::vector<int32_t> schemas;
        if (props && props->t == JVal::Obj) {
            for (const auto& kv : props->obj) {
                std::string enc = encode_json_string(kv.first);
                hashes.push_back(json_key_hash(std::string_view(enc).substr(1, enc.size() - 2)));
                keys.push_back(std::move(enc));
                schemas.push_back(compile(kv.second, depth + 1));
            }
        }
        int32_t additional = kJsAnyNode;
        if (const JVal* ap = v.get("additionalProperties")) {
            if (ap->t == JVal::Bool) additional = (ap->s == "true") ? kJsAnyNode : kJsForbidden;
            else if (ap->t == JVal::Obj) additional = compile(*ap, depth + 1);
        }
        uint64_t required = 0;
        if (const JVal* rq = v.get("required"); rq && rq->t == JVal::Arr) {
            for (const auto& r : rq->arr) {
                if (r.t != JVal::Str) continue;
                const std::string enc = encode_json_string(r.s);
                size_t found = keys.size();
                for (size_t i = 0; i < keys.size(); ++i) if (keys[i] == enc) { found = i; break; }
                if (found == keys.size()) {
                    hashes.push_back(json_key_hash(std::string_view(enc).substr(1, enc.size() - 2)));
                    keys.push_back(enc);
                    schemas.push_back(additional >= 0 ? additional : kJsAnyNode);
                }
                if (found < 64) required |= (1ULL << found);
            }
        }
        int32_t items_idx = kJsAnyNode;
        if (items && items->t == JVal::Obj) items_idx = compile(*items, depth + 1);
        else if (items && items->t == JVal::Bool) items_idx = compile(*items, depth + 1);

        std::vector<std::string> literals;
        if (const JVal* e = v.get("enum"); e && e->t == JVal::Arr && !e->arr.empty()) {
            bool all_scalar = true;
            for (const auto& m : e->arr) {
                std::string ser;
                if (!serialize_scalar(m, &ser)) { all_scalar = false; break; }
                literals.push_back(std::move(ser));
            }
            if (!all_scalar) literals.clear();
        }
        if (const JVal* c = v.get("const")) {
            std::string ser;
            if (serialize_scalar(*c, &ser)) { literals.clear(); literals.push_back(std::move(ser)); }
        }

        auto read_u32 = [&](const char* key, uint32_t* out) {
            const JVal* x = v.get(key);
            if (!x || x->t != JVal::Num || x->s.empty() || x->s[0] == '-') return;
            const unsigned long long u = std::strtoull(x->s.c_str(), nullptr, 10);
            *out = static_cast<uint32_t>(std::min<unsigned long long>(u, UINT32_MAX));
        };
        uint32_t min_len = 0, max_len = UINT32_MAX, min_items = 0, max_items = UINT32_MAX;
        read_u32("minLength", &min_len);
        read_u32("maxLength", &max_len);
        read_u32("minItems", &min_items);
        read_u32("maxItems", &max_items);

        JsonSchemaNode& n = sch.nodes[static_cast<size_t>(idx)];
        n.min_len = min_len;
        n.max_len = max_len;
        n.min_items = min_items;
        n.max_items = max_items;
        n.types = types;
        n.prop_keys = std::move(keys);
        n.prop_hash = std::move(hashes);
        n.prop_schema = std::move(schemas);
        n.required = required;
        n.additional = additional;
        n.items = items_idx;
        n.literals = std::move(literals);
        return idx;
    }

    // Resolves $ref in nodes [first, end); "#" means `doc_root` (the schema document's root).
    void resolve_refs(size_t first, int32_t doc_root) {
        for (size_t ni = first; ni < sch.nodes.size(); ++ni) {
            JsonSchemaNode& n = sch.nodes[ni];
            if (n.ref_path.empty()) continue;
            const std::string r = n.ref_path;
            if (r == "#") { n.ref = doc_root; continue; }
            int32_t target = kJsAnyNode;
            for (const char* key : {"$defs", "definitions"}) {
                const std::string prefix = std::string("#/") + key + "/";
                if (r.compare(0, prefix.size(), prefix) == 0) {
                    auto it = defs.find(std::string(key) + "/" + r.substr(prefix.size()));
                    if (it != defs.end()) target = it->second;
                }
            }
            n.ref = target;
        }
    }
};

// Frame kinds and states.
enum : uint8_t { kValue = 1, kObj = 2, kArr = 3, kStr = 4, kNum = 5, kLit = 6, kLitSet = 7 };
enum : uint8_t { kObjStart = 0, kObjNeedKey = 1, kObjInKey = 2, kObjColon = 3, kObjAfterValue = 4 };
enum : uint8_t { kArrStart = 0, kArrAfterValue = 1, kArrNeedValue = 2 };

// Whitespace between tokens: at most one line break and 40 spaces/tabs per run (enough for
// pretty-printed JSON), and none before the root value. Without a cap the model pads freely.
constexpr uint16_t kWsSpaceCap = 40;
constexpr uint16_t kWsNewline = 0x8000;
constexpr uint32_t kNumMaxLen = 40;
constexpr const char* kLits[3] = {"true", "false", "null"};
constexpr uint32_t kLitLen[3] = {4, 5, 4};
constexpr uint64_t kFnvInit = 1469598103934665603ULL;

inline bool is_ws(uint8_t c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
inline bool is_hex(uint8_t c) { return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
inline uint64_t fnv(uint64_t h, uint8_t c) { return (h ^ c) * 1099511628211ULL; }

// Number states: 0 start, 1 after '-', 2 after leading '0', 3 int digits, 4 after '.',
// 5 frac digits, 6 after e/E, 7 after exp sign, 8 exp digits. Complete: 2, 3, 5, 8.
inline int num_next(uint8_t st, uint8_t c, bool int_only) {
    switch (st) {
        case 0: if (c == '-') return 1; if (c == '0') return 2; if (is_digit(c)) return 3; return -1;
        case 1: if (c == '0') return 2; if (is_digit(c)) return 3; return -1;
        case 2: if (!int_only && c == '.') return 4; if (!int_only && (c == 'e' || c == 'E')) return 6; return -1;
        case 3: if (is_digit(c)) return 3; if (!int_only && c == '.') return 4; if (!int_only && (c == 'e' || c == 'E')) return 6; return -1;
        case 4: if (is_digit(c)) return 5; return -1;
        case 5: if (is_digit(c)) return 5; if (c == 'e' || c == 'E') return 6; return -1;
        case 6: if (c == '+' || c == '-') return 7; if (is_digit(c)) return 8; return -1;
        case 7: if (is_digit(c)) return 8; return -1;
        case 8: if (is_digit(c)) return 8; return -1;
        default: return -1;
    }
}
inline bool num_complete(uint8_t st) { return st == 2 || st == 3 || st == 5 || st == 8; }

inline void set_bit(uint64_t* s, uint32_t i) { s[i >> 6] |= (1ULL << (i & 63)); }
inline bool get_bit(const uint64_t* s, uint32_t i) { return (s[i >> 6] >> (i & 63)) & 1ULL; }

} // namespace

uint64_t json_key_hash(std::string_view encoded_inner) {
    uint64_t h = kFnvInit;
    for (const char c : encoded_inner) h = fnv(h, static_cast<uint8_t>(c));
    return h;
}

void JsonSchema::init_any_object() {
    nodes.clear();
    nodes.emplace_back();  // 0: any
    nodes.emplace_back();  // 1: object of anything
    nodes[1].types = kJsObj;
    root = 1;
}

bool JsonSchema::compile(std::string_view schema_json, std::string* err) {
    nodes.clear();
    nodes.emplace_back();  // 0: any
    JParser jp{schema_json};
    JVal v;
    if (!jp.val(v, 0)) {
        if (err) *err = "response_format schema is not valid JSON";
        return false;
    }
    if (v.t != JVal::Obj && v.t != JVal::Bool) {
        if (err) *err = "response_format schema must be a JSON object";
        return false;
    }
    Compiler c{*this, {}};
    root = kJsAnyNode;
    if (v.t == JVal::Obj) c.collect_defs(v);
    root = c.compile(v, 0);
    c.resolve_refs(0, root);
    return true;
}

void JsonSchema::init_empty() {
    nodes.clear();
    nodes.emplace_back();  // 0: any
    root = kJsAnyNode;
}

int32_t JsonSchema::add_tool(std::string_view name, std::string_view parameters_json) {
    const size_t first = nodes.size();
    int32_t params = kJsAnyNode;
    JParser jp{parameters_json};
    JVal v;
    if (!parameters_json.empty() && jp.val(v, 0) && v.t == JVal::Obj) {
        Compiler c{*this, {}};
        c.collect_defs(v);
        params = c.compile(v, 0);
        c.resolve_refs(first, params);
        // Tool parameters are always an object, even when the schema leaves "type" out.
        JsonSchemaNode& pn = nodes[static_cast<size_t>(resolve(params))];
        if (pn.alts.empty() && pn.literals.empty() && (pn.types & kJsObj)) pn.types = kJsObj;
    } else {
        nodes.emplace_back();
        params = static_cast<int32_t>(nodes.size() - 1);
        nodes[static_cast<size_t>(params)].types = kJsObj;
    }
    nodes.emplace_back();
    const int32_t name_node = static_cast<int32_t>(nodes.size() - 1);
    nodes[static_cast<size_t>(name_node)].types = kJsStr;
    nodes[static_cast<size_t>(name_node)].literals.push_back(encode_json_string(name));

    nodes.emplace_back();
    const int32_t entry = static_cast<int32_t>(nodes.size() - 1);
    JsonSchemaNode& e = nodes[static_cast<size_t>(entry)];
    e.types = kJsObj;
    e.additional = kJsForbidden;
    for (const char* k : {"name", "args"}) {
        std::string enc = encode_json_string(k);
        e.prop_hash.push_back(json_key_hash(std::string_view(enc).substr(1, enc.size() - 2)));
        e.prop_keys.push_back(std::move(enc));
    }
    e.prop_schema = {name_node, params};
    e.required = 3;
    return entry;
}

void JsonSchema::finish_tools(const std::vector<std::string>& names) {
    nodes.emplace_back();
    const int32_t name_node = static_cast<int32_t>(nodes.size() - 1);
    nodes[static_cast<size_t>(name_node)].types = kJsStr;
    for (const auto& nm : names) nodes[static_cast<size_t>(name_node)].literals.push_back(encode_json_string(nm));
    nodes.emplace_back();
    const int32_t args_node = static_cast<int32_t>(nodes.size() - 1);
    nodes[static_cast<size_t>(args_node)].types = kJsObj;
    nodes.emplace_back();
    root = static_cast<int32_t>(nodes.size() - 1);
    JsonSchemaNode& e = nodes[static_cast<size_t>(root)];
    e.types = kJsObj;
    e.additional = kJsForbidden;
    for (const char* k : {"name", "args"}) {
        std::string enc = encode_json_string(k);
        e.prop_hash.push_back(json_key_hash(std::string_view(enc).substr(1, enc.size() - 2)));
        e.prop_keys.push_back(std::move(enc));
    }
    e.prop_schema = {name_node, args_node};
    e.required = 3;
}

int32_t JsonSchema::resolve(int32_t idx) const {
    for (int guard = 0; guard < 32; ++guard) {
        if (idx < 0 || static_cast<size_t>(idx) >= nodes.size()) return kJsAnyNode;
        const int32_t r = nodes[static_cast<size_t>(idx)].ref;
        if (r < 0) return idx;
        idx = r;
    }
    return kJsAnyNode;
}

// ---------------------------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------------------------

void JsonMatcher::copy_stack(Stack& dst, const Stack& src) {
    dst.depth = src.depth;
    dst.ws = src.ws;
    dst.done = src.done;
    std::memcpy(dst.f, src.f, sizeof(Frame) * src.depth);
}

JsonMatcher& JsonMatcher::operator=(const JsonMatcher& o) {
    if (this == &o) return *this;
    schema_ = o.schema_;
    n_ = o.n_;
    started_ = o.started_;
    for (int i = 0; i < n_; ++i) copy_stack(stacks_[i], o.stacks_[i]);
    return *this;
}

void JsonMatcher::init(const JsonSchema* schema) { init(schema, schema->root); }

void JsonMatcher::init(const JsonSchema* schema, int32_t root) {
    schema_ = schema;
    started_ = false;
    n_ = 1;
    Stack& s = stacks_[0];
    s.depth = 1;
    s.ws = 0;
    s.done = false;
    std::memset(&s.f[0], 0, sizeof(Frame));
    s.f[0].kind = kValue;
    s.f[0].node = root;
}

void JsonMatcher::feed(const Stack& in, uint8_t c, Stack* out, int& n_out, int guard) const {
    if (guard > 8 || n_out >= kMaxStacks * 2) return;
    Stack s;
    copy_stack(s, in);
    const JsonSchema& sch = *schema_;

    auto emit = [&](bool reset_ws) {
        if (reset_ws) s.ws = 0;
        if (n_out < kMaxStacks * 2) copy_stack(out[n_out++], s);
    };
    auto ws_step = [&]() {
        if (s.depth == 1 && s.f[0].kind == kValue) return;  // nothing before the root value
        if (c == '\n' || c == '\r') {
            if (c == '\r' || (s.ws & kWsNewline)) {
                // allow "\r\n" as one line break
                if (c == '\r' && !(s.ws & kWsNewline)) { s.ws |= kWsNewline; s.ws |= 0x4000; emit(false); }
                else if (c == '\n' && (s.ws & 0x4000)) { s.ws &= static_cast<uint16_t>(~0x4000); emit(false); }
                return;
            }
            s.ws |= kWsNewline;
            emit(false);
            return;
        }
        if ((s.ws & 0x3FFF) >= kWsSpaceCap) return;
        s.ws++;
        emit(false);
    };
    auto push = [&](uint8_t kind, int32_t node) -> Frame* {
        if (s.depth >= kMaxDepth) return nullptr;
        Frame& f = s.f[s.depth++];
        std::memset(&f, 0, sizeof(Frame));
        f.kind = kind;
        f.node = node;
        return &f;
    };
    auto pop_value = [&]() {
        s.depth--;
        if (s.depth == 0) s.done = true;
    };
    // A key finished: `idx` is the property index or -1 for an additional property.
    auto key_done = [&](int32_t idx) -> bool {
        s.depth--;
        Frame& o = s.f[s.depth - 1];
        const JsonSchemaNode& on = sch.nodes[static_cast<size_t>(sch.resolve(o.node))];
        if (idx < 0 && on.additional == kJsForbidden) return false;
        if (idx >= 0 && idx < 64) {
            if ((o.seen >> idx) & 1ULL) return false;  // duplicate key
            o.seen |= (1ULL << idx);
        }
        o.b = idx;
        o.st = kObjColon;
        return true;
    };

    for (;;) {
        if (s.done || s.depth == 0) return;  // nothing may follow the root value
        Frame& f = s.f[s.depth - 1];
        switch (f.kind) {
        case kValue: {
            if (is_ws(c)) { ws_step(); return; }
            const int32_t ni = sch.resolve(f.node);
            const JsonSchemaNode& n = sch.nodes[static_cast<size_t>(ni)];
            if (!n.alts.empty()) {
                for (const int32_t alt : n.alts) {
                    Stack t;
                    copy_stack(t, s);
                    t.f[t.depth - 1].node = alt;
                    feed(t, c, out, n_out, guard + 1);
                }
                return;
            }
            f.node = ni;
            if (!n.literals.empty()) {
                f.kind = kLitSet; f.aux = 0; f.a = 0;
                const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(n.literals.size(), 256));
                for (uint32_t i = 0; i < cnt; ++i) set_bit(f.set, i);
                continue;
            }
            if (c == '{' && (n.types & kJsObj)) { f.kind = kObj; f.st = kObjStart; f.seen = 0; f.b = -1; emit(true); return; }
            if (c == '[' && (n.types & kJsArr)) { f.kind = kArr; f.st = kArrStart; f.a = 0; emit(true); return; }
            if (c == '"' && (n.types & kJsStr)) { f.kind = kStr; f.st = 0; f.aux = 0; f.a = 0; f.h = kFnvInit; emit(true); return; }
            if ((c == '-' || is_digit(c)) && (n.types & (kJsInt | kJsNum))) {
                f.kind = kNum; f.st = 0; f.a = 0; f.aux = (n.types & kJsNum) ? 0 : 1;
                continue;
            }
            if ((c == 't' || c == 'f') && (n.types & kJsBool)) { f.kind = kLit; f.b = (c == 't') ? 0 : 1; f.a = 0; continue; }
            if (c == 'n' && (n.types & kJsNull)) { f.kind = kLit; f.b = 2; f.a = 0; continue; }
            return;
        }
        case kObj: {
            const JsonSchemaNode& n = sch.nodes[static_cast<size_t>(f.node)];
            if (f.st == kObjStart || f.st == kObjNeedKey) {
                if (is_ws(c)) { ws_step(); return; }
                if (c == '}' && f.st == kObjStart) {
                    if ((n.required & ~f.seen) != 0) return;
                    pop_value(); emit(true); return;
                }
                if (c != '"') return;
                f.st = kObjInKey;
                if (n.additional == kJsForbidden) {
                    const int32_t obj_node = f.node;
                    const uint64_t seen = f.seen;
                    Frame* k = push(kLitSet, obj_node);
                    if (!k) return;
                    k->aux = 1;
                    bool any = false;
                    const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(n.prop_keys.size(), 256));
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (i < 64 && ((seen >> i) & 1ULL)) continue;
                        set_bit(k->set, i);
                        any = true;
                    }
                    if (!any) return;
                    continue;  // feed the opening quote to the key set
                }
                Frame* k = push(kStr, f.node);
                if (!k) return;
                k->aux = 1;
                k->h = kFnvInit;
                emit(true);
                return;
            }
            if (f.st == kObjColon) {
                if (is_ws(c)) { ws_step(); return; }
                if (c != ':') return;
                const int32_t child = (f.b >= 0 && static_cast<size_t>(f.b) < n.prop_schema.size())
                                          ? n.prop_schema[static_cast<size_t>(f.b)]
                                          : (n.additional >= 0 ? n.additional : kJsAnyNode);
                f.st = kObjAfterValue;
                if (!push(kValue, child)) return;
                emit(true);
                return;
            }
            if (f.st == kObjAfterValue) {
                if (is_ws(c)) { ws_step(); return; }
                if (c == ',') {
                    if (n.additional == kJsForbidden) {
                        bool more = false;
                        for (size_t i = 0; i < n.prop_keys.size(); ++i) {
                            if (i >= 64 || !((f.seen >> i) & 1ULL)) { more = true; break; }
                        }
                        if (!more) return;
                    }
                    f.st = kObjNeedKey; emit(true); return;
                }
                if (c == '}') {
                    if ((n.required & ~f.seen) != 0) return;
                    pop_value(); emit(true); return;
                }
                return;
            }
            return;
        }
        case kArr: {
            const JsonSchemaNode& n = sch.nodes[static_cast<size_t>(f.node)];
            if (f.st == kArrStart) {
                if (is_ws(c)) { ws_step(); return; }
                if (c == ']') {
                    if (n.min_items > 0) return;
                    pop_value(); emit(true); return;
                }
                if (n.max_items == 0) return;
                f.st = kArrAfterValue;
                f.a = 1;
                if (!push(kValue, n.items)) return;
                continue;
            }
            if (f.st == kArrAfterValue) {
                if (is_ws(c)) { ws_step(); return; }
                if (c == ',') {
                    if (f.a >= n.max_items) return;
                    f.st = kArrNeedValue; emit(true); return;
                }
                if (c == ']') {
                    if (f.a < n.min_items) return;
                    pop_value(); emit(true); return;
                }
                return;
            }
            f.st = kArrAfterValue;
            f.a++;
            if (!push(kValue, n.items)) return;
            continue;
        }
        case kStr: {
            if (f.st == 0) {
                if (c == '"') {
                    if (f.aux == 1) {
                        const JsonSchemaNode& on = sch.nodes[static_cast<size_t>(f.node)];
                        int32_t idx = -1;
                        for (size_t i = 0; i < on.prop_hash.size(); ++i) {
                            if (on.prop_hash[i] == f.h) { idx = static_cast<int32_t>(i); break; }
                        }
                        if (!key_done(idx)) return;
                    } else {
                        if (f.a < sch.nodes[static_cast<size_t>(f.node)].min_len) return;
                        pop_value();
                    }
                    emit(true);
                    return;
                }
                if (c < 0x20) return;
                // Count code points: a lead byte or an escape's backslash starts one.
                if (f.aux == 0 && (c == '\\' || (c & 0xC0) != 0x80)) {
                    if (f.a >= sch.nodes[static_cast<size_t>(f.node)].max_len) return;
                    f.a++;
                }
                f.h = fnv(f.h, c);
                if (c == '\\') f.st = 1;
                emit(true);
                return;
            }
            if (f.st == 1) {
                if (c == 'u') { f.st = 2; }
                else if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') { f.st = 0; }
                else return;
                f.h = fnv(f.h, c);
                emit(true);
                return;
            }
            if (!is_hex(c)) return;
            f.h = fnv(f.h, c);
            f.st = (f.st >= 5) ? 0 : static_cast<uint8_t>(f.st + 1);
            emit(true);
            return;
        }
        case kNum: {
            const int ns = num_next(f.st, c, f.aux == 1);
            if (ns >= 0) {
                if (++f.a > kNumMaxLen) return;
                f.st = static_cast<uint8_t>(ns);
                emit(true);
                return;
            }
            if (!num_complete(f.st)) return;
            pop_value();
            continue;  // the parent sees this byte
        }
        case kLit: {
            const uint32_t li = static_cast<uint32_t>(f.b);
            if (f.a >= kLitLen[li] || static_cast<uint8_t>(kLits[li][f.a]) != c) return;
            if (++f.a == kLitLen[li]) pop_value();
            emit(true);
            return;
        }
        case kLitSet: {
            const JsonSchemaNode& n = sch.nodes[static_cast<size_t>(f.node)];
            const std::vector<std::string>& cands = (f.aux == 1) ? n.prop_keys : n.literals;
            const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(cands.size(), 256));
            uint64_t nb[4] = {0, 0, 0, 0};
            bool any_next = false;
            bool all_complete = true;
            int32_t complete_idx = -1;
            for (uint32_t i = 0; i < cnt; ++i) {
                if (!get_bit(f.set, i)) continue;
                const std::string& cs = cands[i];
                if (f.a < cs.size() && static_cast<uint8_t>(cs[f.a]) == c) {
                    set_bit(nb, i);
                    any_next = true;
                    if (cs.size() != f.a + 1) all_complete = false;
                    else if (complete_idx < 0) complete_idx = static_cast<int32_t>(i);
                }
            }
            if (any_next) {
                std::memcpy(f.set, nb, sizeof(nb));
                f.a++;
                if (all_complete) {
                    if (f.aux == 1) { if (!key_done(complete_idx)) return; }
                    else pop_value();
                }
                emit(true);
                return;
            }
            // No candidate continues: a candidate that is already complete ends here (numbers).
            int32_t done_idx = -1;
            for (uint32_t i = 0; i < cnt; ++i) {
                if (get_bit(f.set, i) && cands[i].size() == f.a) { done_idx = static_cast<int32_t>(i); break; }
            }
            if (done_idx < 0 || f.aux == 1) return;
            pop_value();
            continue;
        }
        default:
            return;
        }
    }
}

bool JsonMatcher::stack_can_end(const Stack& s) const {
    if (s.done) return true;
    // Only a root-level number or numeric literal can end without a closing byte.
    if (s.depth != 1) return false;
    const Frame& f = s.f[0];
    if (f.kind == kNum) return num_complete(f.st);
    if (f.kind == kLitSet && f.aux == 0) {
        const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
        const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(n.literals.size(), 256));
        for (uint32_t i = 0; i < cnt; ++i) if (get_bit(f.set, i) && n.literals[i].size() == f.a) return true;
    }
    return false;
}

bool JsonMatcher::can_end() const {
    for (int i = 0; i < n_; ++i) if (stack_can_end(stacks_[i])) return true;
    return false;
}

bool JsonMatcher::accepts(std::string_view bytes) const {
    if (bytes.empty()) return false;
    JsonMatcher tmp(*this);
    return tmp.advance(bytes);
}

bool JsonMatcher::step_byte(uint8_t c) {
    Stack out[kMaxStacks * 2];
    int n_out = 0;
    for (int i = 0; i < n_; ++i) feed(stacks_[i], c, out, n_out, 0);
    if (n_out == 0) return false;
    if (n_out > kMaxStacks) n_out = kMaxStacks;
    for (int i = 0; i < n_out; ++i) copy_stack(stacks_[i], out[i]);
    n_ = static_cast<uint8_t>(n_out);
    return true;
}

bool JsonMatcher::advance(std::string_view bytes) {
    if (bytes.empty()) return false;
    JsonMatcher backup(*this);
    for (const char ch : bytes) {
        if (!step_byte(static_cast<uint8_t>(ch))) {
            *this = backup;
            return false;
        }
    }
    started_ = true;
    return true;
}

} // namespace squid
