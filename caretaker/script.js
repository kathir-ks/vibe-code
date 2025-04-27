document.addEventListener('DOMContentLoaded', () => {
    const apiKeyInput = document.getElementById('apiKey');
    const saveApiKeyButton = document.getElementById('saveApiKey');
    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

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
        chatHistory: [] // Store chat history {role: 'user'/'model', parts: [{text: '...'}]}
    };

    function loadData() {
        const storedData = localStorage.getItem(DATA_STORAGE_KEY);
        if (storedData) {
            try {
                appData = JSON.parse(storedData);
                // Ensure all keys exist after loading
                appData.activities = appData.activities || [];
                appData.journalEntries = appData.journalEntries || [];
                appData.tasks = appData.tasks || [];
                appData.interests = appData.interests || [];
                appData.chatHistory = appData.chatHistory || [];
            } catch (error) {
                console.error("Error parsing stored data:", error);
                appData = { activities: [], journalEntries: [], tasks: [], interests: [], chatHistory: [] };
            }
        }
        renderAll(); // Initial render after loading
    }

    function saveData() {
        try {
            // Optional: Limit chat history size to prevent excessive storage use
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
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }

    // Simple HTML escaping function
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
    // (This section remains unchanged)
    const activityInput = document.getElementById('activityInput');
    const addActivityButton = document.getElementById('addActivity');
    const activityList = document.getElementById('activityList');

    function renderActivities() { // Function starts around line 98
        activityList.innerHTML = '';
        if (!appData.activities || appData.activities.length === 0) {
            activityList.innerHTML = '<li>No activities logged yet.</li>';
            return;
        }
        // Line below (~104) is where the previous error seemed related
        [...appData.activities].reverse().forEach(activity => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="timestamp">${activity.timestamp}</span><span class="content">${escapeHtml(activity.description)}</span>`;
            activityList.appendChild(li);
        });
    }

    addActivityButton.addEventListener('click', () => {
        const description = activityInput.value.trim();
        if (description) {
            appData.activities.push({
                timestamp: getCurrentTimestamp(),
                description: description
            });
            saveData();
            renderActivities();
            activityInput.value = '';
        } else {
            alert('Please enter an activity description.');
        }
    });

    // --- Journaling ---
    // (This section remains unchanged)
    const journalInput = document.getElementById('journalInput');
    const addJournalEntryButton = document.getElementById('addJournalEntry');
    const journalList = document.getElementById('journalList');

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
            appData.journalEntries.push({
                timestamp: getCurrentTimestamp(),
                content: content
            });
            saveData();
            renderJournal();
            journalInput.value = '';
        } else {
             alert('Please write something in your journal entry.');
        }
    });

    // --- Task Tracking ---
    // (This section remains unchanged)
    const taskInput = document.getElementById('taskInput');
    const addTaskButton = document.getElementById('addTask');
    const taskList = document.getElementById('taskList');

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
                span.textContent = task.description;
                span.classList.add('task-text');
                if (task.done) {
                    span.classList.add('done');
                }
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
            appData.tasks.push({
                id: generateId(),
                description: description,
                done: false
            });
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
    // (This section remains unchanged)
    const interestInput = document.getElementById('interestInput');
    const addInterestButton = document.getElementById('addInterest');
    const interestList = document.getElementById('interestList');

    function renderInterests() {
        interestList.innerHTML = '';
        if (!appData.interests || appData.interests.length === 0) {
            interestList.innerHTML = '<li>No interests added yet.</li>';
            return;
        }
        appData.interests.forEach((interest, index) => {
            const li = document.createElement('li');
            const span = document.createElement('span');
            span.textContent = interest;
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
    // (Helper functions addMessageToChat, renderChatHistory, prepareContextForAI remain unchanged)
    const chatHistoryDiv = document.getElementById('chatHistory');
    const chatMessageInput = document.getElementById('chatMessage');
    const sendMessageButton = document.getElementById('sendMessage');
    const chatLoading = document.getElementById('chatLoading');
    const chatError = document.getElementById('chatError');

    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'ai-message');
        messageDiv.textContent = message;
        chatHistoryDiv.appendChild(messageDiv);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }

    function renderChatHistory() {
        chatHistoryDiv.innerHTML = '';
        appData.chatHistory.forEach(msg => {
            const text = msg.parts && msg.parts.length > 0 ? msg.parts[0].text : (msg.text || '');
            addMessageToChat(msg.role === 'user' ? 'user' : 'ai', text);
        });
         chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }

    function prepareContextForAI() {
        const MAX_ACTIVITIES = 5;
        const MAX_JOURNAL = 3;
        const MAX_TASKS = 10;

        let context = "--- User Context ---\n";
        if (appData.activities.length > 0) {
            context += "Recent Activities:\n";
            [...appData.activities].slice(-MAX_ACTIVITIES).forEach(a => { context += `- (${a.timestamp}): ${a.description}\n`; });
        } else { context += "Recent Activities: None logged recently.\n"; }

        if (appData.journalEntries.length > 0) {
            context += "\nRecent Journal Entries:\n";
            [...appData.journalEntries].slice(-MAX_JOURNAL).forEach(j => { const snippet = j.content.length > 100 ? j.content.substring(0, 100) + '...' : j.content; context += `- (${j.timestamp}): ${snippet}\n`; });
        } else { context += "\nRecent Journal Entries: None logged recently.\n"; }

        const pendingTasks = appData.tasks.filter(t => !t.done);
        if (pendingTasks.length > 0) {
            context += "\nPending Tasks:\n";
            pendingTasks.slice(0, MAX_TASKS).forEach(t => { context += `- ${t.description}\n`; });
            if (pendingTasks.length > MAX_TASKS) { context += `- ...and ${pendingTasks.length - MAX_TASKS} more.\n`; }
        } else { context += "\nPending Tasks: None\n"; }

        if (appData.interests.length > 0) {
             context += "\nInterests:\n";
             appData.interests.forEach(i => { context += `- ${i}\n`; });
        } else { context += "\nInterests: None listed.\n"; }

        context += "--- End Context ---\n";
        return context;
    }


    // **REVISED & CORRECTED**: Function to call Gemini API
    async function callGeminiApi(userPrompt) { // userPrompt is technically available but we use the history version mostly
        if (!GEMINI_API_KEY) {
            chatError.textContent = 'Error: API Key is not set.';
            chatError.style.display = 'block';
            chatLoading.style.display = 'none';
            return null;
        }

        chatLoading.style.display = 'block';
        chatError.style.display = 'none';

        // Define the system instruction (role definition ONLY)
        const systemInstructionText = `You are a Personal Caretaker assistant. Your objectives are:
1. Help the user track their activities, journal entries, tasks, and interests using the context provided.
2. Engage in supportive and understanding conversations. Help the user maintain good mental health.
3. Be realistic and grounded. Provide a reality check when needed, rather than gaslighting or being overly agreeable if something seems off. Avoid toxic positivity.
4. **You MUST use the context provided in the user's message history to answer questions about activities, journal entries, tasks, and interests.** Do not claim you lack access to information if it is present in the context.
5. Keep your responses concise and helpful unless asked for more detail.`;

        // Prepare the context string
        const contextData = prepareContextForAI();

        // Prepare the chat history for the API
        const MAX_HISTORY_TURNS_FOR_API = 20; // Send last 20 turns (user + ai = 10 pairs)

        // **CORRECTED History Construction**:
        // Get the full history relevant for the API call limit
        const relevantHistory = appData.chatHistory.slice(-MAX_HISTORY_TURNS_FOR_API);

        // Ensure relevantHistory is not empty before trying to access the last element
        if (relevantHistory.length === 0) {
             console.error("Cannot construct API payload: No chat history available.");
             chatError.textContent = 'Error: Cannot send message - chat history is empty.';
             chatError.style.display = 'block';
             chatLoading.style.display = 'none';
             return null;
        }


        // Construct the 'contents' array
        const contents = [
            // 1. Include all historical turns *except* the very last user message.
            //    The slice(0, -1) correctly takes all elements except the last one.
            ...relevantHistory.slice(0, -1),

            // 2. Add the prepared context as a simulated user message.
             {
                role: 'user',
                parts: [{ text: "Okay, remember this is my current situation and context:\n" + contextData }]
            },

            // 3. Add the actual *last* user prompt from the history.
            //    This ensures we send exactly what the user typed most recently.
            relevantHistory[relevantHistory.length - 1] // Access the last item of the sliced history
        ];


        // --- Generation Config, Safety Settings, URL (remain the same) ---
        const generationConfig = {
            temperature: 0.7,
            topK: 1,
            topP: 0.95,
            maxOutputTokens: 350,
        };

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
            console.log("Contents (History + Context + Prompt):", JSON.stringify(contents, null, 2)); // Log the exact structure being sent

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    systemInstruction: {
                        parts: [{ text: systemInstructionText }]
                    },
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

            // --- Response Parsing and Error Handling (remain the same) ---
            let aiResponseText = null;
            if (data.candidates && data.candidates.length > 0) {
                 const candidate = data.candidates[0];
                 if (candidate.finishReason && candidate.finishReason !== "STOP" && candidate.finishReason !== "MAX_TOKENS") {
                     console.error("API response finished unexpectedly:", candidate.finishReason, candidate.safetyRatings);
                     let blockReason = candidate.finishReason;
                     if (candidate.safetyRatings) {
                        const blockedRating = candidate.safetyRatings.find(r => r.blocked);
                        if (blockedRating) {
                            blockReason += ` (Blocked Category: ${blockedRating.category})`;
                        }
                     }
                     throw new Error(`Response generation failed or was blocked. Reason: ${blockReason}`);
                 }
                 if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                    aiResponseText = candidate.content.parts[0].text;
                 }
            }

            if (!aiResponseText) {
                 if (data.promptFeedback && data.promptFeedback.blockReason) {
                     console.error("Response blocked by API prompt filters:", data.promptFeedback);
                     throw new Error(`Response blocked due to prompt ${data.promptFeedback.blockReason}.`);
                 } else {
                    console.error("Unexpected API response structure or empty content:", data);
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
            addMessageToChat('ai', `[Error: Could not get response. ${error.message}]`);
            return null;
        }
    } // --- End of callGeminiApi ---


    // Handle sending chat message
    // (This section remains unchanged)
    sendMessageButton.addEventListener('click', async () => {
        const messageText = chatMessageInput.value.trim();
        if (!messageText) return;
        if (!GEMINI_API_KEY) {
             alert("Please enter and save your Gemini API Key first.");
             return;
        }

        addMessageToChat('user', messageText);
        appData.chatHistory.push({ role: 'user', parts: [{ text: messageText }] });
        saveData();
        chatMessageInput.value = '';

        const aiResponse = await callGeminiApi(messageText); // Pass user prompt (though it's mainly read from history now)

        if (aiResponse) {
            addMessageToChat('ai', aiResponse);
        }
    });

     chatMessageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessageButton.click();
        }
    });

    // --- Initial Load ---
    // (This section remains unchanged)
    function renderAll() {
        renderActivities();
        renderJournal();
        renderTasks();
        renderInterests();
        renderChatHistory();
    }

    loadData(); // Load data and render everything when the page loads

}); // End DOMContentLoaded