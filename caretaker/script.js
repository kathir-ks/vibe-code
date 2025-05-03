document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const apiKeyInput = document.getElementById('apiKey');
    const saveApiKeyButton = document.getElementById('saveApiKey');
    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    const activityInput = document.getElementById('activityInput');
    const addActivityButton = document.getElementById('addActivity');
    const activityList = document.getElementById('activityList');
    const journalInput = document.getElementById('journalInput');
    const addJournalEntryButton = document.getElementById('addJournalEntry');
    const journalList = document.getElementById('journalList');
    const taskInput = document.getElementById('taskInput');
    const addTaskButton = document.getElementById('addTask');
    const taskList = document.getElementById('taskList');
    const interestInput = document.getElementById('interestInput');
    const addInterestButton = document.getElementById('addInterest');
    const interestList = document.getElementById('interestList');
    const chatHistoryDiv = document.getElementById('chatHistory');
    const chatMessageInput = document.getElementById('chatMessage');
    const sendMessageButton = document.getElementById('sendMessage');
    const chatLoading = document.getElementById('chatLoading');
    const chatError = document.getElementById('chatError');
    const clearChatButton = document.getElementById('clearChatButton'); // New button
    const downloadChatButton = document.getElementById('downloadChatButton'); // New button

    // --- API Key Handling ---
    const API_KEY_STORAGE_KEY = 'geminiApiKey';
    let GEMINI_API_KEY = localStorage.getItem(API_KEY_STORAGE_KEY) || '';
    if (GEMINI_API_KEY) {
        apiKeyInput.value = GEMINI_API_KEY;
    }

    saveApiKeyButton.addEventListener('click', () => {
        GEMINI_API_KEY = apiKeyInput.value.trim();
        if (GEMINI_API_KEY) {
            localStorage.setItem(API_KEY_STORAGE_KEY, GEMINI_API_KEY);
            alert('API Key saved locally.');
        } else {
            localStorage.removeItem(API_KEY_STORAGE_KEY);
            alert('API Key cleared.');
        }
    });

    // --- Data Storage ---
    const DATA_STORAGE_KEY = 'personalCaretakerData';
    let appData = {
        activities: [],
        journalEntries: [],
        tasks: [],
        interests: [],
        chatHistory: []
    };

    function loadData() {
        const storedData = localStorage.getItem(DATA_STORAGE_KEY);
        if (storedData) {
            try {
                appData = JSON.parse(storedData);
                // Ensure all keys exist after loading, defaulting to empty arrays
                appData.activities = appData.activities || [];
                appData.journalEntries = appData.journalEntries || [];
                appData.tasks = appData.tasks || [];
                appData.interests = appData.interests || [];
                appData.chatHistory = appData.chatHistory || [];
            } catch (error) {
                console.error("Error parsing stored data:", error);
                // Reset to default if parsing fails
                appData = { activities: [], journalEntries: [], tasks: [], interests: [], chatHistory: [] };
            }
        }
        renderAll(); // Initial render after loading
    }

    function saveData() {
        try {
            // Limit chat history size to prevent excessive storage use
            const MAX_CHAT_HISTORY = 50; // Keep last 50 turns (user + ai)
            if (appData.chatHistory.length > MAX_CHAT_HISTORY) {
                 appData.chatHistory = appData.chatHistory.slice(-MAX_CHAT_HISTORY);
            }
            localStorage.setItem(DATA_STORAGE_KEY, JSON.stringify(appData));
        } catch (error) {
            console.error("Error saving data:", error);
            alert("Could not save data. Local storage might be full.");
        }
    }

    // --- Utilities ---
    function getCurrentTimestamp() {
        return new Date().toLocaleString();
    }

    function generateId() {
        // Simple ID generator
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }

    // Simple HTML escaping function (Corrected Version)
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    // --- Tab Switching Logic ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const targetTab = tab.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // --- Activity Tracking ---
    function renderActivities() {
        activityList.innerHTML = '';
        if (!appData.activities || appData.activities.length === 0) {
            activityList.innerHTML = '<li>No activities logged yet.</li>';
            return;
        }
        [...appData.activities].reverse().forEach(activity => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="timestamp">${activity.timestamp}</span><span class="content">${escapeHtml(activity.description)}</span>`;
            activityList.appendChild(li);
        });
    }

    addActivityButton.addEventListener('click', () => {
        const description = activityInput.value.trim();
        if (description) {
            appData.activities.push({ timestamp: getCurrentTimestamp(), description: description });
            saveData();
            renderActivities();
            activityInput.value = '';
        } else {
            alert('Please enter an activity description.');
        }
    });

    // --- Journaling ---
    function renderJournal() {
        journalList.innerHTML = '';
         if (!appData.journalEntries || appData.journalEntries.length === 0) {
            journalList.innerHTML = '<li>No journal entries yet.</li>';
            return;
        }
        [...appData.journalEntries].reverse().forEach(entry => {
            const li = document.createElement('li');
            li.classList.add('journal-entry');
            li.innerHTML = `<span class="timestamp">${entry.timestamp}</span><span class="content">${escapeHtml(entry.content)}</span>`;
            journalList.appendChild(li);
        });
    }

     addJournalEntryButton.addEventListener('click', () => {
        const content = journalInput.value.trim();
        if (content) {
            appData.journalEntries.push({ timestamp: getCurrentTimestamp(), content: content });
            saveData();
            renderJournal();
            journalInput.value = '';
        } else {
             alert('Please write something in your journal entry.');
        }
    });

    // --- Task Tracking ---
    function renderTasks() {
        taskList.innerHTML = '';
         if (!appData.tasks || appData.tasks.length === 0) {
            taskList.innerHTML = '<li>No tasks added yet.</li>';
            return;
        }
        const pendingTasks = appData.tasks.filter(task => !task.done);
        const doneTasks = appData.tasks.filter(task => task.done);

        const renderTaskList = (tasksToRender) => {
            tasksToRender.forEach((task) => {
                const li = document.createElement('li');
                li.dataset.taskId = task.id;
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.checked = task.done;
                checkbox.addEventListener('change', () => toggleTaskDone(task.id));
                const span = document.createElement('span');
                span.textContent = task.description; // Use textContent for auto-escaping
                span.classList.add('task-text');
                if (task.done) { span.classList.add('done'); }
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Delete';
                deleteButton.classList.add('delete-task');
                deleteButton.addEventListener('click', () => deleteTask(task.id));
                li.appendChild(checkbox);
                li.appendChild(span);
                li.appendChild(deleteButton);
                taskList.appendChild(li);
            });
        };
        renderTaskList(pendingTasks);
        renderTaskList(doneTasks);
    }

    addTaskButton.addEventListener('click', () => {
        const description = taskInput.value.trim();
        if (description) {
            appData.tasks.push({ id: generateId(), description: description, done: false });
            saveData();
            renderTasks();
            taskInput.value = '';
        } else {
             alert('Please enter a task description.');
        }
    });

    function toggleTaskDone(taskId) {
        const task = appData.tasks.find(t => t.id === taskId);
        if (task) {
            task.done = !task.done;
            saveData();
            renderTasks();
        }
    }

     function deleteTask(taskId) {
        appData.tasks = appData.tasks.filter(t => t.id !== taskId);
        saveData();
        renderTasks();
    }

    // --- Interest Tracking ---
    function renderInterests() {
        interestList.innerHTML = '';
        if (!appData.interests || appData.interests.length === 0) {
            interestList.innerHTML = '<li>No interests added yet.</li>';
            return;
        }
        appData.interests.forEach((interest, index) => {
            const li = document.createElement('li');
            const span = document.createElement('span');
            span.textContent = interest; // Use textContent for auto-escaping
            span.classList.add('content');
            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Delete';
            deleteButton.classList.add('delete-interest');
            deleteButton.dataset.interestIndex = index;
            deleteButton.addEventListener('click', deleteInterest);
            li.appendChild(span);
            li.appendChild(deleteButton);
            interestList.appendChild(li);
        });
    }

    addInterestButton.addEventListener('click', () => {
        const interest = interestInput.value.trim();
        if (interest) {
            if (!appData.interests.includes(interest)) {
                 appData.interests.push(interest);
                 saveData();
                 renderInterests();
                 interestInput.value = '';
            } else {
                alert("This interest is already listed.");
            }
        } else {
            alert('Please enter an interest.');
        }
    });

    function deleteInterest(event) {
        const indexToDelete = parseInt(event.target.dataset.interestIndex, 10);
        if (!isNaN(indexToDelete) && indexToDelete >= 0 && indexToDelete < appData.interests.length) {
             appData.interests.splice(indexToDelete, 1);
             saveData();
             renderInterests();
        }
    }

    // --- Chat Functionality ---
    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'ai-message');
        messageDiv.textContent = message; // Use textContent for safe rendering
        chatHistoryDiv.appendChild(messageDiv);
        // Scroll after DOM update
        setTimeout(() => { chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; }, 0);
    }

    function renderChatHistory() {
        chatHistoryDiv.innerHTML = '';
        if (!appData.chatHistory || appData.chatHistory.length === 0) {
            // Optional: Add a placeholder message?
            // const placeholder = document.createElement('div');
            // placeholder.textContent = "Chat history is empty.";
            // placeholder.style.textAlign = 'center';
            // placeholder.style.color = '#888';
            // chatHistoryDiv.appendChild(placeholder);
            return;
        }
        appData.chatHistory.forEach(msg => {
            const text = msg.parts && msg.parts.length > 0 ? msg.parts[0].text : (msg.text || '');
            addMessageToChat(msg.role === 'user' ? 'user' : 'ai', text);
        });
    }

    function prepareContextForAI() {
        const MAX_ACTIVITIES = 5;
        const MAX_JOURNAL = 3;
        const MAX_TASKS = 10;

        let context = "--- User Context ---\n";
        // Activities
        if (appData.activities.length > 0) {
            context += "Recent Activities:\n";
            [...appData.activities].slice(-MAX_ACTIVITIES).forEach(a => { context += `- (${a.timestamp}): ${a.description}\n`; });
        } else { context += "Recent Activities: None logged recently.\n"; }
        // Journal
        if (appData.journalEntries.length > 0) {
            context += "\nRecent Journal Entries:\n";
            [...appData.journalEntries].slice(-MAX_JOURNAL).forEach(j => { const snippet = j.content.length > 100 ? j.content.substring(0, 100) + '...' : j.content; context += `- (${j.timestamp}): ${snippet}\n`; });
        } else { context += "\nRecent Journal Entries: None logged recently.\n"; }
        // Tasks
        const pendingTasks = appData.tasks.filter(t => !t.done);
        if (pendingTasks.length > 0) {
            context += "\nPending Tasks:\n";
            pendingTasks.slice(0, MAX_TASKS).forEach(t => { context += `- ${t.description}\n`; });
            if (pendingTasks.length > MAX_TASKS) { context += `- ...and ${pendingTasks.length - MAX_TASKS} more.\n`; }
        } else { context += "\nPending Tasks: None\n"; }
        // Interests
        if (appData.interests.length > 0) {
             context += "\nInterests:\n";
             appData.interests.forEach(i => { context += `- ${i}\n`; });
        } else { context += "\nInterests: None listed.\n"; }

        context += "--- End Context ---";
        return context;
    }


    // Function to call Gemini API
    async function callGeminiApi() { // Removed userPrompt parameter as it's read from history
        if (!GEMINI_API_KEY) {
            chatError.textContent = 'Error: API Key is not set.';
            chatError.style.display = 'block';
            chatLoading.style.display = 'none';
            return null;
        }

        chatLoading.style.display = 'block';
        chatError.style.display = 'none';

        // Refined System Instruction
        const systemInstructionText = `You are a friendly, supportive, and realistic Personal Caretaker assistant. Your primary goals are:
1.  Help the user track their life via activities, journal entries, tasks, and interests using the CONTEXT BLOCK provided below the conversation history.
2.  Engage in helpful conversations, offering support and understanding for mental well-being.
3.  Be grounded in reality. Offer gentle reality checks if the user seems distressed or unrealistic, but do so kindly. Avoid toxic positivity.
4.  **CRITICAL: You MUST reference the data in the '--- User Context ---' block when relevant.** For example, if asked about tasks, list them from the context. If asked about recent activities, summarize from the context. Acknowledge the user's listed interests. Do not invent information or claim you don't have access if the context provides it.
5.  Keep responses reasonably concise and conversational.`;

        const contextData = prepareContextForAI();
        const MAX_HISTORY_TURNS_FOR_API = 20;
        const relevantHistory = appData.chatHistory.slice(-MAX_HISTORY_TURNS_FOR_API);

        if (relevantHistory.length === 0) {
             console.error("Cannot construct API payload: No chat history available.");
             chatError.textContent = 'Error: Cannot send message - chat history is empty.';
             chatError.style.display = 'block';
             chatLoading.style.display = 'none';
             return null;
        }

        // Construct the 'contents' array
        const contents = [
            ...relevantHistory.slice(0, -1), // History excluding last user prompt
            { // Injected Context
                role: 'user',
                parts: [{ text: "Context for my current request:\n" + contextData }]
            },
            relevantHistory[relevantHistory.length - 1] // Actual last user prompt
        ];

        const generationConfig = { temperature: 0.7, topK: 1, topP: 0.95, maxOutputTokens: 350 };
        const safetySettings = [
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
        ];
        const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${GEMINI_API_KEY}`;

        try {
            console.log("--- Sending API Request ---");
            console.log("System Instruction (Role):", systemInstructionText);
            console.log("Contents (History + Context + Prompt):", JSON.stringify(contents, null, 2));

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    systemInstruction: { parts: [{ text: systemInstructionText }] },
                    contents: contents,
                    generationConfig: generationConfig,
                    safetySettings: safetySettings,
                 }),
            });

            chatLoading.style.display = 'none';

            if (!response.ok) {
                const errorData = await response.json();
                console.error("API Error Response:", errorData);
                throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorData?.error?.message || 'Unknown error'}`);
            }

            const data = await response.json();
            console.log("API Raw Response:", data);
            console.log("--- API Response Received ---");

            let aiResponseText = null;
            if (data.candidates && data.candidates.length > 0) {
                 const candidate = data.candidates[0];
                 if (candidate.finishReason && candidate.finishReason !== "STOP" && candidate.finishReason !== "MAX_TOKENS") {
                     let blockReason = candidate.finishReason;
                     if (candidate.safetyRatings) {
                        const blockedRating = candidate.safetyRatings.find(r => r.blocked);
                        if (blockedRating) blockReason += ` (Blocked Category: ${blockedRating.category})`;
                     }
                     throw new Error(`Response generation failed or was blocked. Reason: ${blockReason}`);
                 }
                 if (candidate.content?.parts?.length > 0) {
                    aiResponseText = candidate.content.parts[0].text;
                 }
            }

            if (!aiResponseText) {
                 if (data.promptFeedback?.blockReason) {
                     throw new Error(`Response blocked due to prompt ${data.promptFeedback.blockReason}.`);
                 } else {
                    throw new Error("Could not parse AI response from API or response was empty.");
                 }
            }

            // Add AI response to history
            appData.chatHistory.push({ role: 'model', parts: [{ text: aiResponseText }] });
            saveData();
            return aiResponseText;

        } catch (error) {
            console.error('Error calling Gemini API:', error);
            chatLoading.style.display = 'none';
            chatError.textContent = `Error: ${error.message}`;
            chatError.style.display = 'block';
            // Add UI error message, but don't save it to history
            addMessageToChat('ai', `[Error: Could not get response. ${error.message}]`);
            return null;
        }
    } // --- End of callGeminiApi ---


    // Handle sending chat message
    async function handleSendMessage() {
        const messageText = chatMessageInput.value.trim();
        if (!messageText) return;
        if (!GEMINI_API_KEY) {
             alert("Please enter and save your Gemini API Key first.");
             return;
        }

        addMessageToChat('user', messageText);
        // Add user message to history *before* sending API call
        appData.chatHistory.push({ role: 'user', parts: [{ text: messageText }] });
        saveData(); // Save immediately
        chatMessageInput.value = ''; // Clear input

        // Get AI response (which will also update history and save on success)
        const aiResponse = await callGeminiApi(); // No need to pass messageText anymore

        if (aiResponse) {
            addMessageToChat('ai', aiResponse);
        }
        // Error display is handled within callGeminiApi
    }

    sendMessageButton.addEventListener('click', handleSendMessage);
    chatMessageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage(); // Call the handler function
        }
    });


    // Event Listener for Clear Chat
    clearChatButton.addEventListener('click', () => {
        if (appData.chatHistory.length === 0) {
            alert("Chat history is already empty.");
            return;
        }
        if (confirm("Are you sure you want to clear the entire chat history? This cannot be undone.")) {
            appData.chatHistory = []; // Clear the array in memory
            saveData();                 // Save the empty array to localStorage
            renderChatHistory();        // Update the UI
            alert("Chat history cleared.");
        }
    });

    // Event Listener for Download Chat
    downloadChatButton.addEventListener('click', () => {
        if (appData.chatHistory.length === 0) {
            alert("Chat history is empty. Nothing to download.");
            return;
        }

        // Format chat history as plain text
        let chatText = `Personal Caretaker Chat History - ${new Date().toLocaleString()}\n`;
        chatText += "==================================================\n\n";

        appData.chatHistory.forEach(msg => {
            const prefix = msg.role === 'user' ? 'User:' : 'AI:';
            const text = msg.parts && msg.parts.length > 0 ? msg.parts[0].text : (msg.text || '');
            // Ensure text content is handled properly, even if empty
            chatText += `${prefix}\n${text || '[empty message]'}\n\n`;
        });

        // Create Blob and Trigger Download
        try {
            const blob = new Blob([chatText], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `personal_caretaker_chat_${Date.now()}.txt`; // Filename with timestamp
            document.body.appendChild(link); // Required for Firefox
            link.click();
            document.body.removeChild(link); // Clean up link element
            URL.revokeObjectURL(url); // Free up memory associated with the object URL
        } catch (error) {
            console.error("Error creating download link:", error);
            alert("Could not initiate chat download.");
        }
    });


    // --- Initial Load ---
    function renderAll() {
        renderActivities();
        renderJournal();
        renderTasks();
        renderInterests();
        renderChatHistory();
    }

    loadData(); // Load data and render everything when the page loads

}); // End DOMContentLoaded