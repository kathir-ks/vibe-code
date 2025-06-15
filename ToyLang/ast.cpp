// Base AST node
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual llvm::Value* codegen(llvm::LLVMContext& context, 
                               llvm::IRBuilder<>& builder, 
                               llvm::Module& module) = 0;
};

// Number literal
class NumberAST : public ASTNode {
    int value;
public:
    NumberAST(int val) : value(val) {}
    llvm::Value* codegen(llvm::LLVMContext& context, 
                        llvm::IRBuilder<>& builder, 
                        llvm::Module& module) override {
        return llvm::ConstantInt::get(context, llvm::APInt(32, value));
    }
};

// Variable reference
class VariableAST : public ASTNode {
    std::string name;
public:
    VariableAST(const std::string& n) : name(n) {}
    llvm::Value* codegen(llvm::LLVMContext& context, 
                        llvm::IRBuilder<>& builder, 
                        llvm::Module& module) override;
};

// Binary operations
class BinaryAST : public ASTNode {
    char op;
    std::unique_ptr<ASTNode> left, right;
public:
    BinaryAST(char o, std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r)
        : op(o), left(std::move(l)), right(std::move(r)) {}
    
    llvm::Value* codegen(llvm::LLVMContext& context, 
                        llvm::IRBuilder<>& builder, 
                        llvm::Module& module) override {
        llvm::Value* L = left->codegen(context, builder, module);
        llvm::Value* R = right->codegen(context, builder, module);
        
        switch (op) {
            case '+': return builder.CreateAdd(L, R, "addtmp");
            case '-': return builder.CreateSub(L, R, "subtmp");
            case '*': return builder.CreateMul(L, R, "multmp");
            case '/': return builder.CreateSDiv(L, R, "divtmp");
            case '<': return builder.CreateICmpSLT(L, R, "cmptmp");
            default: return nullptr;
        }
    }
};

// Function call
class CallAST : public ASTNode {
    std::string callee;
    std::vector<std::unique_ptr<ASTNode>> args;
public:
    CallAST(const std::string& c, std::vector<std::unique_ptr<ASTNode>> a)
        : callee(c), args(std::move(a)) {}
    
    llvm::Value* codegen(llvm::LLVMContext& context, 
                        llvm::IRBuilder<>& builder, 
                        llvm::Module& module) override;
};

// Function definition
class FunctionAST : public ASTNode {
    std::string name;
    std::vector<std::string> args;
    std::unique_ptr<ASTNode> body;
public:
    FunctionAST(const std::string& n, std::vector<std::string> a, 
                std::unique_ptr<ASTNode> b)
        : name(n), args(std::move(a)), body(std::move(b)) {}
    
    llvm::Function* codegenFunction(llvm::LLVMContext& context, 
                                   llvm::IRBuilder<>& builder, 
                                   llvm::Module& module);
};