// Unit tests for pure C++20 squidDGX core primitives (JSON SIMD scanner/escaping, UTF-8 streaming, alignment).

#include "json_constraint.hpp"
#include "tiger.hpp"
#include "tokenizer.hpp"
#include "vram_arena.hpp"
#include <cstdio>
#include <string>
#include <string_view>

static void test_json_escape_and_rollback() {
    char buf[128];
    squid::ByteWriter writer(buf, sizeof(buf));
    const std::string plain = "0123456789abcdef0123456789abc";
    const std::string input = plain + "\"\\\n\x01\xC3\xA9";
    SQUID_ASSERT(writer.try_append_json_escaped(input));
    const std::string expected = "\"" + plain + "\\\"\\\\\\n\\u0001\xC3\xA9\"";
    SQUID_ASSERT(writer.slice() == expected);

    char small[20];
    squid::ByteWriter bounded(small, sizeof(small));
    bounded.append_slice("kept");
    SQUID_ASSERT(!bounded.try_append_json_escaped(plain));
    SQUID_ASSERT(bounded.slice() == "kept");
}

static void test_json_scanner_and_overflow() {
    const std::string_view json = "\"0123456789abcdef\xC3\xA9\\u20ac\\nabc\",";
    char out[64];
    size_t pos = 0;
    std::string_view parsed;
    SQUID_ASSERT(squid::parse_json_string(json, pos, out, sizeof(out), &parsed));
    SQUID_ASSERT(parsed == "0123456789abcdef\xC3\xA9\xE2\x82\xAC\nabc");
    SQUID_ASSERT(json[pos] == ',');

    char small[17];
    pos = 0;
    SQUID_ASSERT(!squid::parse_json_string(json, pos, small, sizeof(small), &parsed));
    SQUID_ASSERT(json[pos] == ',');

    const std::string_view nested = "[\"a longer text with ] and \\\" inside\",{}],";
    pos = 0;
    squid::skip_json_value(nested, pos);
    SQUID_ASSERT(nested[pos] == ',');
}

static void test_thinking_effort_and_alignment() {
    char buf[128];
    squid::ByteWriter w(buf, sizeof(buf));
    SQUID_ASSERT(squid::Tokenizer::try_emit_thinking_effort(w, "medium"));
    SQUID_ASSERT(w.slice() == "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>");

    SQUID_ASSERT(squid::align_up_u64(1, 256) == 256);
    SQUID_ASSERT(squid::align_up_u64(256, 256) == 256);
    SQUID_ASSERT(squid::align_up_u64(257, 256) == 512);
}

static bool json_doc(const squid::JsonSchema& s, const char* doc, bool* complete) {
    squid::JsonMatcher m;
    m.init(&s);
    const bool ok = m.advance(doc);
    *complete = ok && m.can_end();
    return ok;
}

static void test_json_constraint() {
    bool end = false;
    squid::JsonSchema any;
    any.init_any_object();
    SQUID_ASSERT(json_doc(any, "{\"a\": [1, 2.5e3, -0, true, null, \"x\\n\\u00e9\"], \"b\": {}}", &end) && end);
    SQUID_ASSERT(json_doc(any, "{\"a\": [1, 2", &end) && !end);
    SQUID_ASSERT(!json_doc(any, "[1]", &end));
    SQUID_ASSERT(!json_doc(any, "{} x", &end));
    SQUID_ASSERT(!json_doc(any, "{\"a\": 01}", &end));
    SQUID_ASSERT(!json_doc(any, "{\"a\": \"\x01\"}", &end));

    squid::JsonSchema s;
    std::string err;
    SQUID_ASSERT(s.compile(R"({"type":"object","additionalProperties":false,"required":["name","kind","tags","n"],
      "properties":{"name":{"type":"string"},"kind":{"type":"string","enum":["bar","line","pie"]},
      "tags":{"type":"array","items":{"$ref":"#/$defs/Tag"}},"n":{"type":"integer"},"opt":{"type":["number","null"]}},
      "$defs":{"Tag":{"type":"object","additionalProperties":false,"required":["k"],
      "properties":{"k":{"anyOf":[{"type":"string"},{"type":"boolean"}]}}}}})", &err));
    SQUID_ASSERT(json_doc(s, R"({"name":"x","kind":"pie","tags":[{"k":"a"},{"k":true}],"n":-12})", &end) && end);
    SQUID_ASSERT(json_doc(s, R"({"n":3,"tags":[],"kind":"bar","name":"","opt":null})", &end) && end);
    SQUID_ASSERT(!json_doc(s, R"({"name":"x","kind":"pie","tags":[]})", &end));          // missing required n
    SQUID_ASSERT(json_doc(s, R"({"name":"x","kind":"pi)", &end) && !end);                // enum prefix
    SQUID_ASSERT(!json_doc(s, R"({"name":"x","kind":"pi")", &end));                      // enum closed early
    SQUID_ASSERT(!json_doc(s, R"({"zzz":1})", &end));                                     // closed object
    SQUID_ASSERT(!json_doc(s, R"({"name":"x","name":"y")", &end));                        // duplicate key
    SQUID_ASSERT(!json_doc(s, R"({"n":1.5)", &end));                                      // integer
    SQUID_ASSERT(!json_doc(s, R"({"tags":[{"k":1}]})", &end));                            // anyOf
    SQUID_ASSERT(json_doc(s, "{\n  \"name\" : \"x\" ,\n  \"kind\":\"line\"", &end) && !end);

    squid::JsonSchema e;
    SQUID_ASSERT(e.compile(R"({"enum":[1,12,"a"]})", &err));
    SQUID_ASSERT(json_doc(e, "1", &end) && end);
    SQUID_ASSERT(json_doc(e, "12", &end) && end);
    SQUID_ASSERT(!json_doc(e, "13", &end));

    squid::JsonSchema rec;
    SQUID_ASSERT(rec.compile(R"({"type":"object","properties":{"child":{"$ref":"#"}},"additionalProperties":false})", &err));
    SQUID_ASSERT(json_doc(rec, R"({"child":{"child":{}}})", &end) && end);

    squid::JsonMatcher m;
    m.init(&s);
    SQUID_ASSERT(m.advance("{\"na"));
    SQUID_ASSERT(m.accepts("me\":") && !m.accepts("x"));
    SQUID_ASSERT(m.advance("me\":\""));
    SQUID_ASSERT(!s.compile("{not json", &err));
    SQUID_ASSERT(!s.compile("\"{bad\"", &err));
    SQUID_ASSERT(!json_doc(any, " {}", &end));                                   // no leading whitespace
    SQUID_ASSERT(json_doc(any, "{\r\n  \"a\": 1\n}", &end) && end);           // pretty-printed
    SQUID_ASSERT(!json_doc(any, "{\n\n\"a\": 1}", &end));                       // two line breaks in a run
}

int main() {
    test_json_escape_and_rollback();
    test_json_scanner_and_overflow();
    test_thinking_effort_and_alignment();
    test_json_constraint();
    printf("[test_unit] All C++20 unit tests passed!\n");
    return 0;
}
