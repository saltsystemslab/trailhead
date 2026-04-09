#pragma once
// Minimal hand-rolled JSON parser/emitter. No external dependencies.
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace trailhead {

struct JsonValue;
using JsonNull   = std::monostate;
using JsonBool   = bool;
using JsonInt    = int64_t;
using JsonDouble = double;
using JsonString = std::string;
using JsonArray  = std::vector<JsonValue>;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

struct JsonValue {
    using V = std::variant<JsonNull, JsonBool, JsonInt, JsonDouble, JsonString, JsonArray, JsonObject>;
    V v;

    JsonValue()                    : v(JsonNull{}) {}
    JsonValue(std::nullptr_t)      : v(JsonNull{}) {}
    JsonValue(bool b)              : v(JsonBool(b)) {}
    JsonValue(int64_t i)           : v(JsonInt(i)) {}
    JsonValue(int i)               : v(JsonInt(i)) {}
    JsonValue(double d)            : v(JsonDouble(d)) {}
    JsonValue(const std::string& s): v(JsonString(s)) {}
    JsonValue(std::string&& s)     : v(JsonString(std::move(s))) {}
    JsonValue(const char* s)       : v(JsonString(s)) {}
    JsonValue(JsonArray a)         : v(std::move(a)) {}
    JsonValue(JsonObject o)        : v(std::move(o)) {}

    bool is_null()   const { return std::holds_alternative<JsonNull>(v); }
    bool is_bool()   const { return std::holds_alternative<JsonBool>(v); }
    bool is_int()    const { return std::holds_alternative<JsonInt>(v); }
    bool is_double() const { return std::holds_alternative<JsonDouble>(v); }
    bool is_string() const { return std::holds_alternative<JsonString>(v); }
    bool is_array()  const { return std::holds_alternative<JsonArray>(v); }
    bool is_object() const { return std::holds_alternative<JsonObject>(v); }
    bool is_number() const { return is_int() || is_double(); }

    bool               as_bool()   const { return std::get<JsonBool>(v); }
    int64_t            as_int()    const { return std::get<JsonInt>(v); }
    double             as_double() const { return is_double() ? std::get<JsonDouble>(v) : (double)std::get<JsonInt>(v); }
    const std::string& as_string() const { return std::get<JsonString>(v); }
    std::string&       as_string()       { return std::get<JsonString>(v); }
    const JsonArray&   as_array()  const { return std::get<JsonArray>(v); }
    JsonArray&         as_array()        { return std::get<JsonArray>(v); }
    const JsonObject&  as_object() const { return std::get<JsonObject>(v); }
    JsonObject&        as_object()       { return std::get<JsonObject>(v); }

    // Object field lookup — returns nullptr if key missing or not an object
    const JsonValue* get(const std::string& key) const {
        if (!is_object()) return nullptr;
        for (const auto& [k, val] : as_object())
            if (k == key) return &val;
        return nullptr;
    }
    JsonValue* get(const std::string& key) {
        if (!is_object()) return nullptr;
        for (auto& [k, val] : as_object())
            if (k == key) return &val;
        return nullptr;
    }

    const JsonValue& operator[](const std::string& key) const {
        const JsonValue* p = get(key);
        if (!p) throw std::runtime_error("JSON key not found: " + key);
        return *p;
    }
    const JsonValue& operator[](size_t i) const { return as_array().at(i); }

    // Convenience getters with defaults
    std::string get_str(const std::string& key, const std::string& def = "") const {
        const JsonValue* p = get(key);
        if (!p || !p->is_string()) return def;
        return p->as_string();
    }
    int64_t get_int(const std::string& key, int64_t def = 0) const {
        const JsonValue* p = get(key);
        if (!p || !p->is_int()) return def;
        return p->as_int();
    }
    bool get_bool(const std::string& key, bool def = false) const {
        const JsonValue* p = get(key);
        if (!p || !p->is_bool()) return def;
        return p->as_bool();
    }
    std::vector<std::string> get_str_array(const std::string& key) const {
        const JsonValue* p = get(key);
        std::vector<std::string> out;
        if (!p || !p->is_array()) return out;
        for (const auto& el : p->as_array())
            if (el.is_string()) out.push_back(el.as_string());
        return out;
    }

    // Set/insert object field
    JsonValue& set(const std::string& key, JsonValue val) {
        if (!is_object()) v = JsonObject{};
        for (auto& [k, vv] : as_object()) {
            if (k == key) { vv = std::move(val); return *this; }
        }
        as_object().push_back({key, std::move(val)});
        return *this;
    }
};

// ── Parser ────────────────────────────────────────────────────────────────

namespace detail {

struct Parser {
    const char* s;
    size_t len, pos = 0;
    explicit Parser(const std::string& t) : s(t.data()), len(t.size()) {}

    char peek() const { return pos < len ? s[pos] : '\0'; }
    char next()       { return pos < len ? s[pos++] : '\0'; }
    void skip_ws() {
        while (pos < len && (s[pos]==' '||s[pos]=='\t'||s[pos]=='\n'||s[pos]=='\r')) ++pos;
    }
    void expect(char c) {
        skip_ws();
        if (peek() != c) throw std::runtime_error(std::string("JSON: expected '") + c + "' at pos " + std::to_string(pos));
        ++pos;
    }

    std::string parse_string() {
        expect('"');
        std::string r;
        while (pos < len && s[pos] != '"') {
            if (s[pos] == '\\') {
                ++pos;
                char e = next();
                switch (e) {
                    case '"':  r+='"';  break; case '\\': r+='\\'; break;
                    case '/':  r+='/';  break; case 'n':  r+='\n'; break;
                    case 'r':  r+='\r'; break; case 't':  r+='\t'; break;
                    case 'b':  r+='\b'; break; case 'f':  r+='\f'; break;
                    case 'u': {
                        if (pos+4>len) throw std::runtime_error("JSON: bad \\u escape");
                        std::string hex(s+pos,4); pos+=4;
                        unsigned code = (unsigned)std::stoul(hex, nullptr, 16);
                        if (code < 0x80) r += (char)code;
                        else if (code < 0x800) { r += (char)(0xC0|(code>>6)); r += (char)(0x80|(code&0x3F)); }
                        else { r += (char)(0xE0|(code>>12)); r += (char)(0x80|((code>>6)&0x3F)); r += (char)(0x80|(code&0x3F)); }
                        break;
                    }
                    default: r += e;
                }
            } else { r += s[pos++]; }
        }
        expect('"');
        return r;
    }

    JsonValue parse_number() {
        size_t start = pos;
        bool flt = false;
        if (peek()=='-') ++pos;
        while (pos<len && s[pos]>='0' && s[pos]<='9') ++pos;
        if (pos<len && s[pos]=='.') { flt=true; ++pos; while (pos<len && s[pos]>='0' && s[pos]<='9') ++pos; }
        if (pos<len && (s[pos]=='e'||s[pos]=='E')) {
            flt=true; ++pos;
            if (pos<len && (s[pos]=='+'||s[pos]=='-')) ++pos;
            while (pos<len && s[pos]>='0' && s[pos]<='9') ++pos;
        }
        std::string tok(s+start, pos-start);
        if (flt) return JsonValue(std::stod(tok));
        return JsonValue((int64_t)std::stoll(tok));
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c=='"') return parse_string();
        if (c=='{') return parse_object();
        if (c=='[') return parse_array();
        if (c=='t') { pos+=4; return JsonValue(true);    }
        if (c=='f') { pos+=5; return JsonValue(false);   }
        if (c=='n') { pos+=4; return JsonValue(nullptr); }
        if (c=='-' || (c>='0'&&c<='9')) return parse_number();
        throw std::runtime_error(std::string("JSON: unexpected '") + c + "' at pos " + std::to_string(pos));
    }

    JsonValue parse_array() {
        expect('['); skip_ws();
        JsonArray arr;
        if (peek()==']') { ++pos; return arr; }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (peek()==']') { ++pos; break; }
            expect(',');
        }
        return arr;
    }

    JsonValue parse_object() {
        expect('{'); skip_ws();
        JsonObject obj;
        if (peek()=='}') { ++pos; return obj; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws(); expect(':');
            obj.push_back({std::move(key), parse_value()});
            skip_ws();
            if (peek()=='}') { ++pos; break; }
            expect(',');
        }
        return obj;
    }
};

inline std::string esc(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break; case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break; case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << (char)c;
        }
    }
    return o.str();
}

inline void emit(std::ostringstream& o, const JsonValue& v, int ind, int dep) {
    std::string pad(dep * ind, ' ');
    std::string pad1((dep+1) * ind, ' ');
    if (v.is_null())        { o << "null"; }
    else if (v.is_bool())   { o << (v.as_bool() ? "true" : "false"); }
    else if (v.is_int())    { o << v.as_int(); }
    else if (v.is_double()) { o << v.as_double(); }
    else if (v.is_string()) { o << '"' << esc(v.as_string()) << '"'; }
    else if (v.is_array()) {
        const auto& a = v.as_array();
        if (a.empty()) { o << "[]"; return; }
        o << "[\n";
        for (size_t i=0; i<a.size(); ++i) {
            o << pad1; emit(o, a[i], ind, dep+1);
            if (i+1<a.size()) o << ",";
            o << "\n";
        }
        o << pad << "]";
    }
    else if (v.is_object()) {
        const auto& obj = v.as_object();
        if (obj.empty()) { o << "{}"; return; }
        o << "{\n";
        for (size_t i=0; i<obj.size(); ++i) {
            o << pad1 << '"' << esc(obj[i].first) << "\": ";
            emit(o, obj[i].second, ind, dep+1);
            if (i+1<obj.size()) o << ",";
            o << "\n";
        }
        o << pad << "}";
    }
}

} // namespace detail

inline JsonValue json_parse(const std::string& text) {
    detail::Parser p(text);
    return p.parse_value();
}

inline std::string json_emit(const JsonValue& v, int indent = 2) {
    std::ostringstream o;
    detail::emit(o, v, indent, 0);
    o << "\n";
    return o.str();
}

} // namespace trailhead
