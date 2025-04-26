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
                // Reset to default if parsing fails
                appData = { activities: [], journalEntries: [], tasks: [], interests: [], chatHistory: [] };
            }
        }
        renderAll(); // Initial render after loading
    }

    function saveData() {
        try {
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


    // --- Tab Switching Logic ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Deactivate all tabs and content
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Activate clicked tab and corresponding content
            tab.classList.add('active');
            const targetTab = tab.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // --- Activity Tracking ---
    const activityInput = document.getElementById('activityInput');
    const addActivityButton = document.getElementById('addActivity');
    const activityList = document.getElementById('activityList');

    function renderActivities() {
        activityList.innerHTML = ''; // Clear list
        if (!appData.activities || appData.activities.length === 0) {
            activityList.innerHTML = '<li>No activities logged yet.</li>';
            return;
        }
        // Render in reverse chronological order (newest first)
        [...appData.activities].reverse().forEach(activity => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="timestamp">${activity.timestamp}</span><span class="content">${activity.description}</span>`;
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
            activityInput.value = ''; // Clear input
        } else {
            alert('Please enter an activity description.');
        }
    });

    // --- Journaling ---
    const journalInput = document.getElementById('journalInput');
    const addJournalEntryButton = document.getElementById('addJournalEntry');
    const journalList = document.getElementById('journalList');

    function renderJournal() {
        journalList.innerHTML = '';
         if (!appData.journalEntries || appData.journalEntries.length === 0) {
            journalList.innerHTML = '<li>No journal entries yet.</li>';
            return;
        }
        // Newest first
        [...appData.journalEntries].reverse().forEach(entry => {
            const li = document.createElement('li');
            li.classList.add('journal-entry');
            li.innerHTML = `<span class="timestamp">${entry.timestamp}</span><span class="content">${entry.content}</span>`;
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
    const taskInput = document.getElementById('taskInput');
    const addTaskButton = document.getElementById('addTask');
    const taskList = document.getElementById('taskList');

    function renderTasks() {
        taskList.innerHTML = '';
         if (!appData.tasks || appData.tasks.length === 0) {
            taskList.innerHTML = '<li>No tasks added yet.</li>';
            return;
        }
        appData.tasks.forEach((task, index) => {
            const li = document.createElement('li');
            li.dataset.taskId = task.id; // Store ID for easy access

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
            li.appendChild(span); // Text comes after checkbox
            li.appendChild(deleteButton);
            taskList.appendChild(li);
        });
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
            renderTasks(); // Re-render to update style
        }
    }

     function deleteTask(taskId) {
        appData.tasks = appData.tasks.filter(t => t.id !== taskId);
        saveData();
        renderTasks();
    }


    // --- Interest Tracking ---
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
            span.classList.add('content'); // Use content class for consistency

            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Delete';
            deleteButton.classList.add('delete-interest');
            deleteButton.dataset.interestIndex = index; // Store index for deletion
            deleteButton.addEventListener('click', deleteInterest);

            li.appendChild(span);
            li.appendChild(deleteButton);
            interestList.appendChild(li);
        });
    }

    addInterestButton.addEventListener('click', () => {
        const interest = interestInput.value.trim();
        if (interest) {
            // Optional: Prevent duplicates
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
             appData.interests.splice(indexToDelete, 1); // Remove item at index
             saveData();
             renderInterests();
        }
    }


    // --- Chat Functionality ---
    const chatHistoryDiv = document.getElementById('chatHistory');
    const chatMessageInput = document.getElementById('chatMessage');
    const sendMessageButton = document.getElementById('sendMessage');
    const chatLoading = document.getElementById('chatLoading');
    const chatError = document.getElementById('chatError');

    // Function to add a message to the chat display
    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'ai-message');
        messageDiv.textContent = message; // Simple text display
        chatHistoryDiv.appendChild(messageDiv);
        // Scroll to the bottom
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }

    // Function to render the entire chat history
    function renderChatHistory() {
        chatHistoryDiv.innerHTML = '';
        appData.chatHistory.forEach(msg => {
            // The API response structure nests text under 'parts'
            const text = msg.parts && msg.parts.length > 0 ? msg.parts[0].text : (msg.text || '[empty message]');
            addMessageToChat(msg.role === 'user' ? 'user' : 'ai', text);
        });
    }

    // Function to call Gemini API
    async function callGeminiApi(prompt) {
        if (!GEMINI_API_KEY) {
            chatError.textContent = 'Error: API Key is not set.';
            chatError.style.display = 'block';
            chatLoading.style.display = 'none';
            return null;
        }

        chatLoading.style.display = 'block'; // Show loading indicator
        chatError.style.display = 'none'; // Hide previous errors

        // --- Prompt Engineering ---
        // Basic system instruction + user prompt
        // More advanced: Include summaries of recent activities/journal/tasks here
        // This history format is crucial for the API
        const history = [...appData.chatHistory]; // Copy current history
        const contents = [
            ...history, // Previous turns
            { // Current user turn
                role: 'user',
                parts: [{ text: prompt }],
            },
        ];

        const generationConfig = {
            temperature: 0.7, // Adjust creativity vs factualness
            topK: 1,
            topP: 1,
            maxOutputTokens: 256, // Limit response length
        };

        const safetySettings = [ // Standard safety settings
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
        ];

        // IMPORTANT: Use 'gemini-1.5-flash-latest' or another available model
        const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${GEMINI_API_KEY}`;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: contents, // Pass the constructed conversation history
                    generationConfig: generationConfig,
                    safetySettings: safetySettings,
                    // You could add a systemInstruction here for overall guidance:
                    // systemInstruction: { parts: [{ text: "You are a helpful personal caretaker. Be supportive but realistic. Do not gaslight. Refer to user's logged data if provided in the context."}] },
                 }),
            });

            chatLoading.style.display = 'none'; // Hide loading indicator

            if (!response.ok) {
                const errorData = await response.json();
                console.error("API Error Response:", errorData);
                throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorData?.error?.message || 'Unknown error'}`);
            }

            const data = await response.json();

            // Extract the response text
            if (data.candidates && data.candidates.length > 0 && data.candidates[0].content && data.candidates[0].content.parts && data.candidates[0].content.parts.length > 0) {
                const aiResponseText = data.candidates[0].content.parts[0].text;
                 // Add AI response to history (using the API's structure)
                appData.chatHistory.push({ role: 'model', parts: [{ text: aiResponseText }] });
                saveData(); // Save updated history
                return aiResponseText;
            } else if (data.promptFeedback && data.promptFeedback.blockReason) {
                 // Handle blocked responses due to safety settings
                 console.error("Response blocked:", data.promptFeedback);
                 throw new Error(`Response blocked due to ${data.promptFeedback.blockReason}.`);
            }
             else {
                console.error("Unexpected API response structure:", data);
                throw new Error("Could not parse AI response from API.");
            }

        } catch (error) {
            console.error('Error calling Gemini API:', error);
            chatLoading.style.display = 'none';
            chatError.textContent = `Error: ${error.message}`;
            chatError.style.display = 'block';
            return null; // Indicate failure
        }
    }

    // Handle sending chat message
    sendMessageButton.addEventListener('click', async () => {
        const messageText = chatMessageInput.value.trim();
        if (!messageText) return; // Don't send empty messages
        if (!GEMINI_API_KEY) {
             alert("Please enter and save your Gemini API Key first.");
             return;
        }


        // Display user message immediately
        addMessageToChat('user', messageText);
         // Add user message to history before sending API call
         appData.chatHistory.push({ role: 'user', parts: [{ text: messageText }] });
         saveData(); // Save user message


        chatMessageInput.value = ''; // Clear input field

        // Get AI response
        const aiResponse = await callGeminiApi(messageText); // Pass only the new message

        if (aiResponse) {
            addMessageToChat('ai', aiResponse);
            // AI response already added to appData.chatHistory inside callGeminiApi
        }
    });

     // Allow sending message with Enter key
     chatMessageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { // Check for Enter key without Shift
            e.preventDefault(); // Prevent default newline in input
            sendMessageButton.click(); // Trigger button click
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