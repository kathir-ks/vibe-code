int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <source.toy>\n";
        return 1;
    }
    
    // Read source file
    std::ifstream file(argv[1]);
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    // Compile
    Lexer lexer(source);
    Parser parser(lexer);
    CodeGenerator codegen;
    
    std::vector<std::unique_ptr<FunctionAST>> functions;
    
    // Parse all functions
    while (true) {
        if (auto func = parser.parseDefinition()) {
            functions.push_back(std::move(func));
        } else {
            break;
        }
    }
    
    // Generate code
    codegen.generateCode(functions);
    codegen.createExecutable("output");
    
    std::cout << "Compilation successful! Run with ./output\n";
    return 0;
}