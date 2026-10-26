// SPDX-License-Identifier: MPL-2.0

#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace gemm::json
{
    Value Value::makeBool(bool value)
    {
        Value result;
        result.m_type = Type::Bool;
        result.m_bool = value;
        return result;
    }

    Value Value::makeNumber(double value)
    {
        Value result;
        result.m_type = Type::Number;
        result.m_number = value;
        return result;
    }

    Value Value::makeString(std::string value)
    {
        Value result;
        result.m_type = Type::String;
        result.m_string = std::move(value);
        return result;
    }

    Value Value::makeArray(Array value)
    {
        Value result;
        result.m_type = Type::Array;
        result.m_array = std::move(value);
        return result;
    }

    Value Value::makeObject(Object value)
    {
        Value result;
        result.m_type = Type::Object;
        result.m_object = std::move(value);
        return result;
    }

    bool Value::asBool() const
    {
        if(!isBool())
            throw std::runtime_error("JSON value is not a boolean");
        return m_bool;
    }

    double Value::asNumber() const
    {
        if(!isNumber())
            throw std::runtime_error("JSON value is not a number");
        return m_number;
    }

    std::string const& Value::asString() const
    {
        if(!isString())
            throw std::runtime_error("JSON value is not a string");
        return m_string;
    }

    Value::Array const& Value::asArray() const
    {
        if(!isArray())
            throw std::runtime_error("JSON value is not an array");
        return m_array;
    }

    Value::Object const& Value::asObject() const
    {
        if(!isObject())
            throw std::runtime_error("JSON value is not an object");
        return m_object;
    }

    Value const& Value::at(std::string_view key) const
    {
        auto const& object = asObject();
        auto const found = object.find(key);
        if(found == object.end())
            throw std::out_of_range("JSON object has no key '" + std::string{key} + "'");
        return found->second;
    }

    bool Value::contains(std::string_view key) const
    {
        if(!isObject())
            return false;
        auto const& object = asObject();
        return object.find(key) != object.end();
    }

    Value const& Value::at(std::size_t index) const
    {
        auto const& array = asArray();
        if(index >= array.size())
            throw std::out_of_range("JSON array index out of range");
        return array[index];
    }

    std::size_t Value::size() const
    {
        if(isArray())
            return m_array.size();
        if(isObject())
            return m_object.size();
        return 0u;
    }

    namespace
    {
        void appendEscaped(std::string& out, std::string_view text)
        {
            for(char const c : text)
            {
                switch(c)
                {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if(static_cast<unsigned char>(c) < 0x20u)
                    {
                        char buffer[7];
                        std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
                        out += buffer;
                    }
                    else
                    {
                        out += c;
                    }
                }
            }
        }

        void appendNumber(std::string& out, double value)
        {
            if(!std::isfinite(value))
            {
                out += "null";
                return;
            }
            char buffer[64];
            auto const result = std::to_chars(buffer, buffer + sizeof(buffer), value);
            out.append(buffer, result.ptr);
        }

        void serializeInto(std::string& out, Value const& value, int indentWidth, int depth)
        {
            auto const indent = [&](int level)
            {
                if(indentWidth <= 0)
                    return;
                out.append(static_cast<std::size_t>(level * indentWidth), ' ');
            };

            switch(value.type())
            {
            case Value::Type::Null:
                out += "null";
                break;
            case Value::Type::Bool:
                out += value.asBool() ? "true" : "false";
                break;
            case Value::Type::Number:
                appendNumber(out, value.asNumber());
                break;
            case Value::Type::String:
                out += '"';
                appendEscaped(out, value.asString());
                out += '"';
                break;
            case Value::Type::Array:
                {
                    auto const& array = value.asArray();
                    if(array.empty())
                    {
                        out += "[]";
                        break;
                    }
                    out += '[';
                    if(indentWidth > 0)
                        out += '\n';
                    for(std::size_t i = 0; i < array.size(); ++i)
                    {
                        indent(depth + 1);
                        serializeInto(out, array[i], indentWidth, depth + 1);
                        if(i + 1u < array.size())
                            out += ',';
                        if(indentWidth > 0)
                            out += '\n';
                    }
                    indent(depth);
                    out += ']';
                    break;
                }
            case Value::Type::Object:
                {
                    auto const& object = value.asObject();
                    if(object.empty())
                    {
                        out += "{}";
                        break;
                    }
                    out += '{';
                    if(indentWidth > 0)
                        out += '\n';
                    std::size_t i = 0;
                    for(auto const& [key, child] : object)
                    {
                        indent(depth + 1);
                        out += '"';
                        appendEscaped(out, key);
                        out += "\":";
                        if(indentWidth > 0)
                            out += ' ';
                        serializeInto(out, child, indentWidth, depth + 1);
                        if(++i < object.size())
                            out += ',';
                        if(indentWidth > 0)
                            out += '\n';
                    }
                    indent(depth);
                    out += '}';
                    break;
                }
            }
        }
    } // namespace

    std::string escape(std::string_view text)
    {
        std::string result;
        appendEscaped(result, text);
        return result;
    }

    std::string serialize(Value const& value, int indentWidth)
    {
        std::string result;
        serializeInto(result, value, indentWidth, 0);
        result += '\n';
        return result;
    }

    namespace detail
    {
        char Parser::peek() const
        {
            if(atEnd())
                fail("unexpected end of JSON input");
            return m_text[m_pos];
        }

        char Parser::take()
        {
            char const c = peek();
            ++m_pos;
            return c;
        }

        void Parser::skipSpace()
        {
            while(!atEnd())
            {
                char const c = m_text[m_pos];
                if(c != ' ' && c != '\t' && c != '\n' && c != '\r')
                    break;
                ++m_pos;
            }
        }

        void Parser::expect(char expected)
        {
            if(take() != expected)
                fail(std::string{"expected '"} + expected + "'");
        }

        [[noreturn]] void Parser::fail(std::string const& message) const
        {
            throw std::runtime_error("JSON parse error at byte " + std::to_string(m_pos) + ": " + message);
        }

        std::string Parser::parseString()
        {
            expect('"');
            std::string result;
            while(true)
            {
                if(atEnd())
                    fail("unterminated string");
                char const c = take();
                if(c == '"')
                    break;
                if(c != '\\')
                {
                    result += c;
                    continue;
                }
                if(atEnd())
                    fail("unterminated escape");
                switch(take())
                {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case '/':
                    result += '/';
                    break;
                case 'b':
                    result += '\b';
                    break;
                case 'f':
                    result += '\f';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    fail("unsupported escape sequence");
                }
            }
            return result;
        }

        Value Parser::parseNumber()
        {
            std::size_t const begin = m_pos;
            if(peek() == '-')
                ++m_pos;
            while(!atEnd())
            {
                char const c = m_text[m_pos];
                if((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
                    ++m_pos;
                else
                    break;
            }
            double value = 0.0;
            auto const* first = m_text.data() + begin;
            auto const* last = m_text.data() + m_pos;
            auto const result = std::from_chars(first, last, value, std::chars_format::general);
            if(result.ec != std::errc{} || result.ptr != last)
                fail("malformed number");
            return Value::makeNumber(value);
        }

        Value Parser::parseArray()
        {
            expect('[');
            Value::Array array;
            skipSpace();
            if(!atEnd() && peek() == ']')
            {
                take();
                return Value::makeArray(std::move(array));
            }
            while(true)
            {
                skipSpace();
                array.push_back(parseValue());
                skipSpace();
                char const separator = take();
                if(separator == ']')
                    break;
                if(separator != ',')
                    fail("expected ',' or ']'");
            }
            return Value::makeArray(std::move(array));
        }

        Value Parser::parseObject()
        {
            expect('{');
            Value::Object object;
            skipSpace();
            if(!atEnd() && peek() == '}')
            {
                take();
                return Value::makeObject(std::move(object));
            }
            while(true)
            {
                skipSpace();
                auto key = parseString();
                skipSpace();
                expect(':');
                skipSpace();
                object.emplace(std::move(key), parseValue());
                skipSpace();
                char const separator = take();
                if(separator == '}')
                    break;
                if(separator != ',')
                    fail("expected ',' or '}'");
            }
            return Value::makeObject(std::move(object));
        }

        Value Parser::parseValue()
        {
            skipSpace();
            char const c = peek();
            switch(c)
            {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
                return Value::makeString(parseString());
            case 't':
                if(m_text.substr(m_pos, 4) == "true")
                {
                    m_pos += 4;
                    return Value::makeBool(true);
                }
                break;
            case 'f':
                if(m_text.substr(m_pos, 5) == "false")
                {
                    m_pos += 5;
                    return Value::makeBool(false);
                }
                break;
            case 'n':
                if(m_text.substr(m_pos, 4) == "null")
                {
                    m_pos += 4;
                    return Value{};
                }
                break;
            default:
                break;
            }
            if(c == '-' || (c >= '0' && c <= '9'))
                return parseNumber();
            fail("unexpected character");
        }
    } // namespace detail

    Value parse(std::string_view text)
    {
        detail::Parser parser{text};
        Value value = parser.parseValue();
        return value;
    }
} // namespace gemm::json
