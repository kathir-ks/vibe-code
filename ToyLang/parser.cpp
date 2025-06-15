class Parser {
    Lexer& lexer;
    Token currentToken;
    
public:
    Parser(Lexer& l) : lexer(l) {
        currentToken = lexer.getNextToken();
    }
    
    std::unique_ptr<ASTNode> parseExpression() {
        auto left = parsePrimary();
        return parseBinaryOpRHS(0, std::move(left));
    }
    
    std::unique_ptr<ASTNode> parsePrimary() {
        switch (currentToken.type) {
            case TOK_IDENTIFIER:
                return parseIdentifierExpr();
            case TOK_NUMBER:
                return parseNumber();
            case '(':
                return parseParenExpr();
            default:
                return nullptr;
        }
    }
    
    std::unique_ptr<ASTNode> parseNumber() {
        auto result = std::make_unique<NumberAST>(currentToken.numValue);
        getNextToken(); // consume number
        return result;
    }
    
    std::unique_ptr<FunctionAST> parseDefinition() {
        getNextToken(); // consume 'def'
        
        if (currentToken.type != TOK_IDENTIFIER) return nullptr;
        
        std::string fnName = currentToken.value;
        getNextToken();
        
        if (currentToken.type != '(') return nullptr;
        
        // Parse arguments
        std::vector<std::string> argNames;
        getNextToken();
        while (currentToken.type == TOK_IDENTIFIER) {
            argNames.push_back(currentToken.value);
            getNextToken();
            if (currentToken.type == ',') getNextToken();
        }
        
        if (currentToken.type != ')') return nullptr;
        getNextToken();
        
        // Parse body
        auto body = parseExpression();
        if (!body) return nullptr;
        
        return std::make_unique<FunctionAST>(fnName, std::move(argNames), 
                                           std::move(body));
    }
    
private:
    void getNextToken() { currentToken = lexer.getNextToken(); }
};