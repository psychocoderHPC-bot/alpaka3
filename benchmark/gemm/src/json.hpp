// SPDX-License-Identifier: MPL-2.0
//
// Small, dependency-free JSON reader/writer surface used by the GEMM benchmark
// skeleton. It intentionally supports only the subset needed to load the
// workload manifest and to emit samples.json; it is not a general JSON library.

#pragma once

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gemm::json
{
    class Value
    {
    public:
        using Array = std::vector<Value>;
        using Object = std::map<std::string, Value, std::less<>>;

        enum class Type
        {
            Null,
            Bool,
            Number,
            String,
            Array,
            Object
        };

        Value() = default;

        static Value makeBool(bool value);
        static Value makeNumber(double value);
        static Value makeString(std::string value);
        static Value makeArray(Array value);
        static Value makeObject(Object value);

        [[nodiscard]] Type type() const
        {
            return m_type;
        }

        [[nodiscard]] bool isNull() const
        {
            return m_type == Type::Null;
        }

        [[nodiscard]] bool isBool() const
        {
            return m_type == Type::Bool;
        }

        [[nodiscard]] bool isNumber() const
        {
            return m_type == Type::Number;
        }

        [[nodiscard]] bool isString() const
        {
            return m_type == Type::String;
        }

        [[nodiscard]] bool isArray() const
        {
            return m_type == Type::Array;
        }

        [[nodiscard]] bool isObject() const
        {
            return m_type == Type::Object;
        }

        [[nodiscard]] bool asBool() const;
        [[nodiscard]] double asNumber() const;
        [[nodiscard]] std::string const& asString() const;
        [[nodiscard]] Array const& asArray() const;
        [[nodiscard]] Object const& asObject() const;

        // Object lookup. Missing keys throw std::out_of_range.
        [[nodiscard]] Value const& at(std::string_view key) const;
        [[nodiscard]] bool contains(std::string_view key) const;

        // Array lookup. Out-of-range indexes throw std::out_of_range.
        [[nodiscard]] Value const& at(std::size_t index) const;
        [[nodiscard]] std::size_t size() const;

    private:
        Type m_type{Type::Null};
        bool m_bool{false};
        double m_number{0.0};
        std::string m_string;
        Array m_array;
        Object m_object;
    };

    // Throws std::runtime_error with a byte offset on malformed input.
    [[nodiscard]] Value parse(std::string_view text);

    [[nodiscard]] std::string escape(std::string_view text);

    [[nodiscard]] std::string serialize(Value const& value, int indentWidth = 2);

    namespace detail
    {
        class Parser
        {
        public:
            explicit Parser(std::string_view text) : m_text{text}
            {
            }

            Value parseValue();

        private:
            [[nodiscard]] bool atEnd() const
            {
                return m_pos >= m_text.size();
            }

            [[nodiscard]] char peek() const;
            char take();
            void skipSpace();
            void expect(char expected);
            [[nodiscard]] std::string parseString();
            [[nodiscard]] Value parseNumber();
            [[nodiscard]] Value parseArray();
            [[nodiscard]] Value parseObject();
            [[noreturn]] void fail(std::string const& message) const;

            std::string_view m_text;
            std::size_t m_pos{0};
        };
    } // namespace detail
} // namespace gemm::json
