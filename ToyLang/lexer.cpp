// Token types
enum TokenType {
    TOK_EOF = -1,
    TOK_DEF = -2,
    TOK_IDENTIFIER = -3,
    TOK_NUMBER = -4,
    TOK_IF = -5,
    TOK_THEN = -6,
    TOK_ELSE = -7,
    TOK_RETURN = -8,
    TOK_LET = -9,
    TOK_PRINT = -10
};

struct Token {
    TokenType type;
    std::string value;
    int numValue;
};

class Lexer {
private:
    std::string input;
    size_t pos = 0;
    
public:
    Lexer(const std::string& src) : input(src) {}
    
    Token getNextToken() {
        // Skip whitespace
        while (pos < input.size() && isspace(input[pos])) pos++;
        
        if (pos >= input.size()) return {TOK_EOF, "", 0};
        
        // Numbers
        if (isdigit(input[pos])) {
            std::string numStr;
            while (pos < input.size() && isdigit(input[pos])) {
                numStr += input[pos++];
            }
            return {TOK_NUMBER, numStr, std::stoi(numStr)};
        }
        
        // Identifiers and keywords
        if (isalpha(input[pos])) {
            std::string ident;
            while (pos < input.size() && (isalnum(input[pos]) || input[pos] == '_')) {
                ident += input[pos++];
            }
            
            // Check for keywords
            if (ident == "def") return {TOK_DEF, ident, 0};
            if (ident == "if") return {TOK_IF, ident, 0};
            if (ident == "else") return {TOK_ELSE, ident, 0};
            if (ident == "return") return {TOK_RETURN, ident, 0};
            if (ident == "let") return {TOK_LET, ident, 0};
            if (ident == "print") return {TOK_PRINT, ident, 0};
            
            return {TOK_IDENTIFIER, ident, 0};
        }
        
        // Single character tokens
        char c = input[pos++];
        return {static_cast<TokenType>(c), std::string(1, c), 0};
    }
};