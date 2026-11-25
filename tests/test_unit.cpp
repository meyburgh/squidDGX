// Unit tests for pure C++20 squidDGX core primitives (JSON SIMD scanner/escaping, UTF-8 streaming, alignment).

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

int main() {
    test_json_escape_and_rollback();
    test_json_scanner_and_overflow();
    test_thinking_effort_and_alignment();
    printf("[test_unit] All C++20 unit tests passed!\n");
    return 0;
}
