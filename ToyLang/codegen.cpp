class CodeGenerator {
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::Module> module;
    std::map<std::string, llvm::Value*> namedValues;
    
public:
    CodeGenerator() : builder(context) {
        module = std::make_unique<llvm::Module>("ToyLang", context);
    }
    
    void generateCode(std::vector<std::unique_ptr<FunctionAST>>& functions) {
        // Generate code for all functions
        for (auto& func : functions) {
            func->codegenFunction(context, builder, *module);
        }
        
        // Print the generated IR
        module->print(llvm::errs(), nullptr);
    }
    
    void createExecutable(const std::string& filename) {
        // Initialize LLVM
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();
        
        // Get target triple
        std::string targetTriple = llvm::sys::getDefaultTargetTriple();
        module->setTargetTriple(targetTriple);
        
        // Create target machine
        std::string error;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        
        if (!target) {
            llvm::errs() << error;
            return;
        }
        
        llvm::TargetMachine* targetMachine = target->createTargetMachine(
            targetTriple, "generic", "", llvm::TargetOptions(), 
            llvm::Reloc::PIC_);
        
        module->setDataLayout(targetMachine->createDataLayout());
        
        // Generate object file
        std::error_code EC;
        llvm::raw_fd_ostream dest(filename + ".o", EC, llvm::sys::fs::OF_None);
        
        if (EC) {
            llvm::errs() << "Could not open file: " << EC.message();
            return;
        }
        
        llvm::legacy::PassManager pass;
        if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, 
                                             llvm::CGFT_ObjectFile)) {
            llvm::errs() << "TargetMachine can't emit a file of this type";
            return;
        }
        
        pass.run(*module);
        dest.flush();
        
        // Link with system linker (platform-specific)
        std::string linkCmd = "gcc " + filename + ".o -o " + filename;
        system(linkCmd.c_str());
    }
};