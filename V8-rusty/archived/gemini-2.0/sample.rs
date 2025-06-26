// Copyright 2014 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// src/execution/execution.h

// Modified to Rust

#![allow(dead_code)] // Suppress warnings for now
#![allow(unused_variables)] // Suppress warnings for unused variables
#![allow(non_snake_case)]  //allow non snake case vars

use std::ptr;

// Placeholder imports and definitions (replace with actual Rust types)
mod api_inl { // mock api-inl
    pub struct Local<T>(T); // mock v8::Local
}
mod debug { // mock debug
    pub struct DisableBreak; // mock debug::DisableBreak
    impl DisableBreak {
        pub fn new(_debug: &Debug) -> Self {Self}
    }
    pub struct Debug;
}
mod execution {
    pub struct SaveContext<'a>(&'a super::internal::Isolate);
    impl<'a> SaveContext<'a> {
        pub fn new(isolate: &'a super::internal::Isolate) -> Self {
            SaveContext(isolate)
        }
    }

    pub struct AllowJavascriptExecution;
    impl AllowJavascriptExecution {
        pub fn is_allowed(_isolate: &super::internal::Isolate) -> bool {
            true // mock implementation
        }
    }

    pub struct ThrowOnJavascriptExecution;
    impl ThrowOnJavascriptExecution {
        pub fn is_allowed(_isolate: &super::internal::Isolate) -> bool {
            true // mock implementation
        }
    }

    pub struct DumpOnJavascriptExecution;
    impl DumpOnJavascriptExecution {
        pub fn is_allowed(_isolate: &super::internal::Isolate) -> bool {
            true // mock implementation
        }
    }
}
mod frames {
    // mock execution::frames
}
mod isolate_inl {
    //mock execution::isolate_inl
}
mod logging {
    pub struct RCS_SCOPE<'a>(&'a super::internal::Isolate, RuntimeCallCounterId);
    impl<'a> RCS_SCOPE<'a> {
        pub fn new(isolate: &'a super::internal::Isolate, id: RuntimeCallCounterId) -> Self {
            RCS_SCOPE(isolate, id)
        }
    }
    pub enum RuntimeCallCounterId {
        kInvoke,
        kJS_Execution,
    }
}
mod vm_state_inl {
    pub struct VMState<T>(T);
    impl<T> VMState<T> {
        pub fn new(_state: T) -> Self { Self( _state )}
    }

    pub enum JS {} // mock JS type
}

mod compiler {
    pub enum CWasmEntryParameters { // mock compiler::CWasmEntryParameters
        kCodeEntry,
        kObjectRef,
        kArgumentsBuffer,
        kCEntryFp,
    }
}
mod wasm {
    pub struct WasmCodeManager; //mock wasm::WasmCodeManager
    impl WasmCodeManager {
        pub fn has_memory_protection_key_support(&self) -> bool { false } // mock impl
        pub fn memory_protection_key_writable(&self) -> bool { false } // mock impl
    }
    pub fn get_wasm_code_manager() -> &'static WasmCodeManager {
        static MANAGER: WasmCodeManager = WasmCodeManager {};
        &MANAGER
    }
}
mod trap_handler {
    pub fn set_thread_in_wasm() {} // mock impl
    pub fn is_thread_in_wasm() -> bool {false} //mock impl
    pub fn clear_thread_in_wasm() {} // mock impl
}

mod base {
    pub struct Vector<T>(Vec<T>); //mock base::Vector
    impl<T> Vector<T> {
        pub fn new() -> Self {Vector(Vec::new())}
        pub fn from_vec(vec: Vec<T>) -> Self {Vector(vec)}
        pub fn size(&self) -> usize { self.0.len() }
        pub fn get(&self, index: usize) -> &T {
            &self.0[index]
        }
        pub fn data(&self) -> *const T {
            self.0.as_ptr()
        }
        pub fn push(&mut self, value: T) {
            self.0.push(value)
        }
    }
}
mod message_template {
    pub enum MessageTemplate {
        kNoSideEffectDebugEvaluate,
        kVarRedeclaration,
    }
}
mod utils { // mock utils
    use super::api_inl::Local;

    pub fn open_direct_handle<T>(obj: &T) -> DirectHandle<T> {
        DirectHandle(*obj)
    }

    pub fn to_local<T>(context: &super::internal::NativeContext) -> Local<super::internal::NativeContext> {
        Local(*context)
    }
}

mod script_context_table {
    use super::internal::*;

    #[derive(Debug, Clone, Copy)]
    pub struct ScriptContextTable;

    impl ScriptContextTable {
        pub fn add(
            isolate: &Isolate,
            script_context: &DirectHandle<ScriptContextTable>,
            result: &DirectHandle<Context>,
            ignore_duplicates: bool,
        ) -> DirectHandle<ScriptContextTable> {
            // Implement the logic here to add to the script context table
            todo!("ScriptContextTable::add")
        }

        pub fn lookup(
            &self,
            name: &DirectHandle<String>,
            lookup: &mut VariableLookupResult,
        ) -> bool {
            // Implement the logic here to lookup in the script context table
            todo!("ScriptContextTable::lookup")
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct VariableLookupResult {
        pub mode: VariableMode,
        pub context_index: usize, //mock for context index
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VariableMode {
        Const, //mock variable mode
        Let, //mock variable mode
    }
}

mod lookup_iterator {
    use super::internal::*;

    #[derive(Debug, Clone, Copy)]
    pub struct LookupIterator {
        // Implement the fields here
    }

    impl LookupIterator {
        pub const OWN_SKIP_INTERCEPTOR: i32 = 0; //mock constant
        pub fn new(
            isolate: &Isolate,
            global_object: &DirectHandle<JSGlobalObject>,
            name: &DirectHandle<String>,
            global_object2: &DirectHandle<JSGlobalObject>,
            own_skip_interceptor: i32,
        ) -> Self {
            // Implement the logic here to create the lookup iterator
            todo!("LookupIterator::new")
        }
    }
}

mod scope_info {
    use super::internal::*;

    #[derive(Debug, Clone, Copy)]
    pub struct ScopeInfo {
        // Implement the fields here
    }

    impl ScopeInfo {
        pub fn iterate_local_names(scope_info: &DirectHandle<ScopeInfo>) -> LocalNamesIterator {
            // Implement the logic here to iterate local names
            todo!("ScopeInfo::iterate_local_names")
        }

        pub fn context_local_mode(&self, index: usize) -> VariableMode {
            // Implement the logic here to get the context local mode
            todo!("ScopeInfo::context_local_mode")
        }

        pub fn is_repl_mode_scope(&self) -> bool {
            // Implement the logic here to check if it's a REPL mode scope
            todo!("ScopeInfo::is_repl_mode_scope")
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct LocalNamesIterator {
        // Implement the fields here
    }

    impl LocalNamesIterator {
        pub fn name(&self) -> DirectHandle<String> {
            // Implement the logic here to get the name
            todo!("LocalNamesIterator::name")
        }

        pub fn index(&self) -> usize {
            // Implement the logic here to get the index
            todo!("LocalNamesIterator::index")
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VariableMode {
        Const, //mock variable mode
        Let, //mock variable mode
    }
}

pub mod v8 {
    pub type Isolate = i32; // Mock v8::Isolate
    pub struct TryCatch(_isolate: i32); //Mock v8::TryCatch
    impl TryCatch {
        pub fn new(isolate: *mut Isolate) -> TryCatch { TryCatch(isolate as i32) }
        pub fn set_verbose(&mut self, _verbose: bool) {}
        pub fn set_capture_message(&mut self, _capture_message: bool) {}
        pub fn has_caught(&self) -> bool {false} //mock return
        pub fn exception(&self) -> *mut i32 { ptr::null_mut() } //mock return
    }

    pub struct Platform; // Mock v8::Platform
    pub fn get_current_platform() -> &'static Platform {
        static PLATFORM: Platform = Platform;
        &PLATFORM
    }
    impl Platform {
        pub fn dump_without_crashing(&self) {} // mock impl
    }

    pub type Context = i32; //Mock v8::Context
    pub type AbortScriptExecutionCallback = unsafe extern "C" fn(
        isolate: *mut Isolate,
        context: *mut Context,
    );

}

pub mod internal {
    use super::*; // Import parent scope

    use std::convert::TryInto;
    use std::mem::size_of;

    // Placeholder types (replace with actual Rust types)
    #[derive(Debug, Clone, Copy)]
    pub struct Object;

    #[derive(Debug, Clone, Copy)]
    pub struct JSFunction;

    #[derive(Debug, Clone, Copy)]
    pub struct JSGlobalObject;

    #[derive(Debug, Clone, Copy)]
    pub struct JSGlobalProxy;

    #[derive(Debug, Clone, Copy)]
    pub struct Code;

    #[derive(Debug, Clone, Copy)]
    pub struct Heap;

    #[derive(Debug, Clone, Copy)]
    pub struct IsolateData;

    #[derive(Debug, Clone, Copy)]
    pub struct Context;

    #[derive(Debug, Clone, Copy)]
    pub struct NativeContext;

    #[derive(Debug, Clone, Copy)]
    pub struct FixedArray;

    #[derive(Debug, Clone, Copy)]
    pub struct FunctionTemplateInfo;

    #[derive(Debug, Clone, Copy)]
    pub struct Script;

    #[derive(Debug, Clone, Copy)]
    pub struct ScopeInfo;

    #[derive(Debug, Clone, Copy)]
    pub struct String;

    #[derive(Debug, Clone, Copy)]
    pub struct MicrotaskQueue;

    #[derive(Debug, Clone, Copy)]
    pub struct ScriptContextTable;

    pub type Address = usize;
    #[derive(Debug, Clone, Copy)]
    pub struct DirectHandle<T>(pub T);

    #[derive(Debug, Clone, Copy)]
    pub struct IndirectHandle<T>(pub T); // Mock IndirectHandle

    #[derive(Debug, Clone, Copy)]
    pub struct MaybeDirectHandle<T>(pub Option<T>);

    impl<T> MaybeDirectHandle<T> {
        pub fn is_null(&self) -> bool {
            self.0.is_none()
        }
        pub fn to_handle(&self) -> Result<DirectHandle<T>, ()> {
            match self.0 {
                Some(val) => Ok(DirectHandle(val)),
                None => Err(()),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct WasmCodePointer {
        value: Address
    }

    impl WasmCodePointer {
        pub fn value(&self) -> Address {
            self.value
        }
    }

    pub struct Factory {
        //mock factory
    }
    impl Factory {
        pub fn undefined_value(&self) -> DirectHandle<Object> { DirectHandle(Object) } // mock impl
        pub fn the_hole_value(&self) -> DirectHandle<Object> { DirectHandle(Object) } // mock impl
        pub fn new_eval_error(&self, _template: message_template::MessageTemplate) -> DirectHandle<Object> { DirectHandle(Object) } // mock impl
        pub fn new_syntax_error(&self, _template: message_template::MessageTemplate, _name: &DirectHandle<String>) -> DirectHandle<Object> { DirectHandle(Object) } // mock impl
        pub fn new_script_context(&self, _native_context: &DirectHandle<NativeContext>, _scope_info: &DirectHandle<ScopeInfo>) -> DirectHandle<Context> { DirectHandle(Context) } // mock impl
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Isolate {
        // Add necessary fields for Isolate
    }

    impl Isolate {
        pub fn factory(&self) -> &Factory {
            static FACTORY: Factory = Factory{};
            &FACTORY
        }

        pub fn native_context(&self) -> DirectHandle<NativeContext> {
            DirectHandle(NativeContext) //mock return
        }
        pub fn debug(&self) -> &debug::Debug {
            static DEBUG: debug::Debug = debug::Debug;
            &DEBUG
        }
        pub fn isolate_data(&self) -> &IsolateData {
            static ISOLATE_DATA: IsolateData = IsolateData {};
            &ISOLATE_DATA
        }

        pub fn clear_exception(&self) {} // mock impl
        pub fn has_exception(&self) -> bool { false } // mock impl
        pub fn set_exception(&self, _exception: Tagged<Object>) {} // mock impl
        pub fn should_check_side_effects(&self) -> bool { false } // mock impl
        pub fn throw(&self, _object: Object) {} // mock impl
        pub fn throw_at(&self, _object: DirectHandle<Object>, _location: &MessageLocation) {} // mock impl
        pub fn report_pending_messages(&self, _report: bool) {} // mock impl
        pub fn increment_javascript_execution_counter(&self) {} // mock impl
        pub fn is_execution_terminating(&self) -> bool { false } // mock impl
        pub fn throw_illegal_operation(&self) {} // mock impl
        pub fn c_entry_fp_address(&self) -> *mut Address { todo!("c_entry_fp_address") } //mock impl. Return a raw pointer. Replace `Address` with the appropriate type.
        pub fn js_entry_sp_address(&self) -> *mut Address { todo!("js_entry_sp_address") } //mock impl. Return a raw pointer. Replace `Address` with the appropriate type.

        pub fn thread_local_top(&self) -> &ThreadLocalTop { todo!("thread_local_top") } //mock impl
    }

    pub struct ThreadLocalTop {
        pub handler_: Address
    }

    pub fn direct_handle<T>(obj: T, _isolate: &Isolate) -> DirectHandle<T> {
        DirectHandle(obj)
    }
    pub fn indirect_handle<T>(obj: &DirectHandle<T>, _isolate: &Isolate) -> IndirectHandle<T> { // mock
        IndirectHandle(obj.0)
    }

    pub fn tag<T>(obj: &T) -> Tagged<T> {
        Tagged(*obj)
    }

    impl NativeContext {
        pub fn global_object(&self) -> JSGlobalObject { JSGlobalObject {} } // mock impl
        pub fn script_context_table(&self) -> ScriptContextTable { ScriptContextTable {} } // mock impl
        pub fn synchronized_set_script_context_table(&self, _table: ScriptContextTable) {} // mock impl

        pub fn script_execution_callback(&self) -> Object { Object {} } // mock impl

    }

    impl JSFunction {
        pub fn shared(&self) -> &SharedFunctionInfo {
            static SFI: SharedFunctionInfo = SharedFunctionInfo{};
            &SFI
        }
        pub fn code(&self, _isolate: &Isolate) -> &Code {
            static CODE: Code = Code{};
            &CODE
        }
        pub fn context(&self) -> &Context {
            static CONTEXT: Context = Context{};
            &CONTEXT
        }

        pub fn set_context(&self, _context: Context) {} // mock

    }

    impl Script {
        //mock script structure
    }

    impl Code {
        pub fn instruction_start(&self) -> Address { 0 } // mock impl
        pub fn is_builtin(&self) -> bool { false } //mock impl
    }

    impl ScriptContextTable {
        pub fn add(_isolate: &Isolate, _script_context: &DirectHandle<ScriptContextTable>, _result: &DirectHandle<Context>, _ignore_duplicates: bool) -> DirectHandle<ScriptContextTable> {
            DirectHandle(ScriptContextTable {})
        }

        pub fn lookup(
            &self,
            name: &DirectHandle<String>,
            lookup: &mut VariableLookupResult,
        ) -> bool {
            // Implement the logic here to lookup in the script context table
            todo!("ScriptContextTable::lookup_impl")
        }
    }

    impl Context {
        pub fn scope_info(&self) -> &ScopeInfo {
            static SCOPE_INFO: ScopeInfo = ScopeInfo{};
            &SCOPE_INFO
        }

        pub fn initialize(&self, _isolate: &Isolate) {} // mock impl

    }

    #[derive(Debug, Clone, Copy)]
    pub struct SharedFunctionInfo;

    impl SharedFunctionInfo {
        pub fn is_script(&self) -> bool { false } //mock impl
        pub fn needs_script_context(&self) -> bool { false } //mock impl
        pub fn api_func_data(&self) -> FunctionTemplateInfo { FunctionTemplateInfo {} } // mock impl
        pub fn BreakAtEntry(&self, _isolate: &Isolate) -> bool { false } // mock impl
        pub fn IsApiFunction(&self) -> bool { false } //mock impl
        pub fn scope_info(&self) -> &ScopeInfo {
            static SCOPE_INFO: ScopeInfo = ScopeInfo{};
            &SCOPE_INFO
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Tagged<T>(pub T);

    impl Heap {
        //implement heap struct
    }

    impl IsolateData {
        pub fn isolate_root(&self) -> Address { 0 } //mock impl
    }

    impl String {
        //mock string struct
    }

    impl MicrotaskQueue {
        //mock MicrotaskQueue struct
    }

    pub struct StackLimitCheck(_isolate: &Isolate); // mock
    impl StackLimitCheck {
        pub fn has_overflowed(&self) -> bool { false } // mock
    }

    pub struct SaveAndSwitchContext<'a>(&'a Isolate, &'a Context); // mock
    impl<'a> SaveAndSwitchContext<'a> {
        pub fn new(isolate: &'a Isolate, context: &'a Context) -> Self { SaveAndSwitchContext(isolate, context) }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct MessageLocation(pub Script, pub i32, pub i32); //mock structure

    // Mock builtins
    pub mod Builtins {
        use super::*;

        pub fn invoke_api_function(
            _isolate: &Isolate,
            _is_construct: bool,
            _fun_data: DirectHandle<FunctionTemplateInfo>,
            _receiver: DirectHandle<Object>,
            _args: base::Vector<const DirectHandle<Object>>,
            _new_target: &HeapObject,
        ) -> MaybeHandle<Object> {
            MaybeHandle(None) // Mock implementation
        }
    }

    pub enum JSParameterCount {
        _0,
    }
    impl From<usize> for JSParameterCount {
        fn from(count: usize) -> Self {
            match count {
                _ => Self::_0, //mock conversion. Add specific cases for other arities
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct HeapObject; //mock HeapObject

    pub fn is_js_global_object(_object: &Object) -> bool { false } // mock impl
    pub fn is_fixed_array(_object: &Object) -> bool { false } //mock impl
    pub fn is_js_function(_object: &Object) -> bool { false } //mock impl
    pub fn is_constructor(_function: &JSFunction) -> bool { false } //mock impl
    pub fn is_js_global_proxy(_object: &Object) -> bool { false } //mock impl
    pub fn is_undefined(_object: Object, _isolate: &Isolate) -> bool { false } //mock impl
    pub fn is_exception(_value: Tagged<Object>, _isolate: &Isolate) -> bool { false } //mock impl
    pub enum MessageHandling {
        kReport,
        kKeepPending,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub enum Target {
        kCallable,
        kRunMicrotasks,
    }

    // Mock Builtin codes
    pub enum BuiltinCode {
        JSConstructEntry,
        JSEntry,
        JSRunMicrotasksEntry,
    }

    pub fn builtin_code(_isolate: &Isolate, builtin: BuiltinCode) -> DirectHandle<Code> {
        DirectHandle(Code) //mock return
    }

    // Stack handler constats mock
    pub mod StackHandlerConstants {
        pub const kNextOffset: usize = 0;
        pub const kPaddingOffset: usize = 8; // Assuming Address is 8 bytes (64-bit)
        pub const kSize: usize = 16;
    }

    pub fn get_current_stack_position() -> Address { 0 } // mock
    pub const K_NULL_ADDRESS: Address = 0;
    pub mod JSGlobalObject {
        use super::*;

        pub fn invalidate_property_cell(_global_object: &DirectHandle<JSGlobalObject>, _name: &DirectHandle<String>) {} // mock impl
    }

    pub mod JSReceiver {
        use super::*;
        pub fn get_property_attributes(_lookup_it: &LookupIterator) -> Result<PropertyAttributes, ()> {
            // Implement the logic here to get the property attributes
            todo!("JSReceiver::get_property_attributes")
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum PropertyAttributes { //mock enum
        DONT_DELETE
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VariableMode { //mock enum
        LEXICAL_VARIABLE_MODE
    }

    pub fn is_lexical_variable_mode(mode: VariableMode) -> bool {
        // Implement the logic here to check if it's a lexical variable mode
        todo!("is_lexical_variable_mode")
    }
}

mod flags {
    //mock commandline flags
    pub struct Flags {
        pub clear_exceptions_on_js_entry: bool,
        pub verify_heap: bool,
        pub strict_termination_checks: bool
    }
    impl Flags {
        pub const fn new() -> Self {
            Flags {
                clear_exceptions_on_js_entry: false,
                verify_heap: false,
                strict_termination_checks: false
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref V8_FLAGS: flags::Flags = flags::Flags::new();
}

pub mod wasm_compiler {
    // mock compiler::wasm_compiler module
}

pub mod Execution {
    use super::*;

    use std::convert::TryInto;

    fn normalize_receiver(
        isolate: &internal::Isolate,
        receiver: internal::DirectHandle<internal::Object>,
    ) -> internal::DirectHandle<internal::Object> {
        // Convert calls on global objects to be calls on the global
        // receiver instead to avoid having a 'this' pointer which refers
        // directly to a global object.
        if internal::is_js_global_object(&receiver.0) {
            let global_proxy = internal::direct_handle(
                internal::JSGlobalObject {},
                isolate,
            );
            internal::direct_handle(global_proxy.0, isolate)
        } else {
            receiver
        }
    }

    struct InvokeParams {
        target: internal::DirectHandle<internal::Object>,
        receiver: internal::DirectHandle<internal::Object>,
        args: base::Vector<const internal::DirectHandle<internal::Object>>,
        new_target: internal::DirectHandle<internal::Object>,
        microtask_queue: *mut internal::MicrotaskQueue,

        message_handling: internal::MessageHandling,
        exception_out: *mut internal::MaybeDirectHandle<internal::Object>,

        is_construct: bool,
        execution_target: internal::Target,
    }

    impl InvokeParams {
        fn set_up_for_new(
            isolate: &internal::Isolate,
            constructor: internal::DirectHandle<internal::Object>,
            new_target: internal::DirectHandle<internal::Object>,
            args: base::Vector<const internal::DirectHandle<internal::Object>>,
        ) -> InvokeParams {
            InvokeParams {
                target: constructor,
                receiver: isolate.factory().undefined_value(),
                args,
                new_target,
                microtask_queue: ptr::null_mut(),
                message_handling: internal::MessageHandling::kReport,
                exception_out: ptr::null_mut(),
                is_construct: true,
                execution_target: internal::Target::kCallable,
            }
        }

        fn set_up_for_call(
            isolate: &internal::Isolate,
            callable: internal::DirectHandle<internal::Object>,
            receiver: internal::DirectHandle<internal::Object>,
            args: base::Vector<const internal::DirectHandle<internal::Object>>,
        ) -> InvokeParams {
            let receiver = normalize_receiver(isolate, receiver);
            InvokeParams {
                target: callable,
                receiver,
                args,
                new_target: isolate.factory().undefined_value(),
                microtask_queue: ptr::null_mut(),
                message_handling: internal::MessageHandling::kReport,
                exception_out: ptr::null_mut(),
                is_construct: false,
                execution_target: internal::Target::kCallable,
            }
        }

        fn set_up_for_try_call(
            isolate: &internal::Isolate,
            callable: internal::DirectHandle<internal::Object>,
            receiver: internal::DirectHandle<internal::Object>,
            args: base::Vector<const internal::DirectHandle<internal::Object>>,
            message_handling: internal::MessageHandling,
            exception_out: *mut internal::MaybeDirectHandle<internal::Object>,
        ) -> InvokeParams {
            let receiver = normalize_receiver(isolate, receiver);
            InvokeParams {
                target: callable,
                receiver,
                args,
                new_target: isolate.factory().undefined_value(),
                microtask_queue: ptr::null_mut(),
                message_handling,
                exception_out,
                is_construct: false,
                execution_target: internal::Target::kCallable,
            }
        }

        fn set_up_for_run_microtasks(
            isolate: &internal::Isolate,
            microtask_queue: *mut internal::MicrotaskQueue,
        ) -> InvokeParams {
            let undefined = isolate.factory().undefined_value();
            InvokeParams {
                target: undefined,
                receiver: undefined,
                args: base::Vector::new(),
                new_target: undefined,
                microtask_queue,
                message_handling: internal::MessageHandling::kReport,
                exception_out: ptr::null_mut(),
                is_construct: false,
                execution_target: internal::Target::kRunMicrotasks,
            }
        }

        fn is_script(&self) -> bool {
            if !internal::is_js_function(&self.target.0) {
                return false;
            }
            let function = unsafe { &*(self.target.0 as *const internal::JSFunction) };
            function.shared().is_script()
        }

        fn get_and_reset_host_defined_options(&mut self) -> internal::DirectHandle<internal::FixedArray> {
            assert!(self.is_script());
            assert_eq!(self.args.size(), 1);
            let options = unsafe { *(self.args.get(0) as *const internal::DirectHandle<internal::Object>) }; // mock
            self.args = base::Vector::new();
            internal::direct_handle(internal::FixedArray, &super::internal::Isolate{}) //mock fixed array
        }
    }

    fn js_entry(
        isolate: &internal::Isolate,
        execution_target: internal::Target,
        is_construct: bool,
    ) -> internal::DirectHandle<internal::Code> {
        if is_construct {
            assert_eq!(internal::Target::kCallable, execution_target);
            internal::builtin_code(isolate, internal::BuiltinCode::JSConstructEntry)
        } else if execution_target == internal::Target::kCallable {
            assert!(!is_construct);
            internal::builtin_code(isolate, internal::BuiltinCode::JSEntry)
        } else if execution_target == internal::Target::kRunMicrotasks {
            assert!(!is_construct);
            internal::builtin_code(isolate, internal::BuiltinCode::JSRunMicrotasksEntry)
        } else {
            panic!("Unreachable");
        }
    }

    fn new_script_context(
        isolate: &internal::Isolate,
        function: internal::DirectHandle<internal::JSFunction>,
        host_defined_options: internal::DirectHandle<internal::FixedArray>,
    ) -> internal::MaybeDirectHandle<internal::Context> {
        // TODO(cbruni, 1244145): Use passed in host_defined_options.
        // Creating a script context is a side effect, so abort if that's not
        // allowed.
        if isolate.should_check_side_effects() {
            isolate.throw(isolate.factory().new_eval_error(
                message_template::MessageTemplate::kNoSideEffectDebugEvaluate,
            ).0);
            return internal::MaybeDirectHandle(None);
        }

        execution::SaveContext::new(isolate); //mock saving context
        //let save = SaveAndSwitchContext(isolate, function.context());
        let sfi = function.0.shared();
        let script = internal::direct_handle(internal::Script, isolate);
        let scope_info = internal::direct_handle(sfi.scope_info().clone(), isolate);
        let native_context = internal::direct_handle(internal::NativeContext, isolate);
        let global_object = internal::direct_handle(native_context.0.global_object(), isolate);
        let script_context = internal::direct_handle(native_context.0.script_context_table(), isolate);

        // Find name clashes.
        for name_it in scope_info::ScopeInfo::iterate_local_names(&scope_info) {
            let name = name_it.name();
            let mode = scope_info.0.context_local_mode(name_it.index());
            let mut lookup = script_context_table::VariableLookupResult {
                mode: script_context_table::VariableMode::Const, //mock
                context_index: 0 //mock
            };

            if script_context.0.lookup(&name, &mut lookup) {
                if internal::is_lexical_variable_mode(mode) || internal::is_lexical_variable_mode(lookup.mode) {
                    let context = internal::direct_handle(internal::Context, isolate); //mock context
                    // If we are trying to redeclare a REPL-mode let as a let, REPL-mode
                    // const as a const, REPL-mode using as a using and REPL-mode await
                    // using as an await using allow it.
                    if !((mode == lookup.mode && internal::is_lexical_variable_mode(mode)) &&
                        scope_info.0.is_repl_mode_scope() &&
                        context.0.scope_info().is_repl_mode_scope()) {
                        // ES#sec-globaldeclarationinstantiation 5.b:
                        // If envRec.HasLexicalDeclaration(name) is true, throw a SyntaxError
                        // exception.
                        let location = MessageLocation(script.0, 0, 1);
                        isolate.throw_at(isolate.factory().new_syntax_error(
                            message_template::MessageTemplate::kVarRedeclaration,
                            &name,
                        ), &location);
                        return internal::MaybeDirectHandle(None);
                    }
                }
            }

            if internal::is_lexical_variable_mode(mode) {
                let lookup_it = lookup_iterator::LookupIterator::new(
                    isolate,
                    &global_object,
                    &name,
                    &global_object,
                    lookup_iterator::LookupIterator::OWN_SKIP_INTERCEPTOR,
                );
                let maybe = internal::JSReceiver::get_property_attributes(&lookup_it);
                // Can't fail since the we looking up own properties on the global object
                // skipping interceptors.
                if maybe.is_err() {
                    panic!();
                }
                let maybe = maybe.unwrap();
                if (maybe as i32 & internal::PropertyAttributes::DONT_DELETE as i32) != 0 {
                    // ES#sec-globaldeclarationinstantiation 5.a:
                    // If envRec.HasVarDeclaration(name) is true, throw a SyntaxError
                    // exception.
                    // ES#sec-globaldeclarationinstantiation 5.d:
                    // If hasRestrictedGlobal is true, throw a SyntaxError exception.
                    let location = MessageLocation(script.0, 0, 1);
                    isolate.throw_at(isolate.factory().new_syntax_error(
                        message_template::MessageTemplate::kVarRedeclaration,
                        &name,
                    ), &location);
                    return internal::MaybeDirectHandle(None);
                }

                internal::JSGlobalObject::invalidate_property_cell(&global_object, &name);