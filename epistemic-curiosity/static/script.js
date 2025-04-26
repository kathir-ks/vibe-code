const postsContainer = document.getElementById('posts-container');
const loadingIndicator = document.getElementById('loading-indicator');

// API endpoint remains relative
const API_ENDPOINT = '/generate-post';

let isLoading = false;
let initialLoadCount = 1; // Start with 1 post initially due to longer generation time

// --- fetchPost function remains the same ---
async function fetchPost() {
    if (isLoading) return;
    isLoading = true;
    loadingIndicator.classList.remove('hidden');
    console.log("Fetching new long-form post...");

    try {
        const response = await fetch(API_ENDPOINT); // Default timeout might be hit here eventually
        if (!response.ok) {
            let errorMsg = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch (jsonError) {/* Ignore */}
            // Handle specific timeout gateway errors if possible
            if (response.status === 504) {
                 errorMsg = "Content generation timed out. The topic might have been too complex or the request took too long. Please try scrolling again for a new topic.";
            }
            throw new Error(errorMsg);
        }
        const data = await response.json();
        console.log("Received post data (topic):", data.topic); // Avoid logging huge content
        displayPost(data);
    } catch (error) {
        console.error('Error fetching post:', error);
        displayError(error.message);
    } finally {
        isLoading = false;
        // Keep indicator potentially longer if needed, but hide eventually
         setTimeout(() => {
             if (!isLoading) {
                 loadingIndicator.classList.add('hidden');
             }
         }, 500); // Slightly longer delay
    }
}


// --- Modified displayPost function using marked.js ---
function displayPost(postData) {
    if (!postData || !postData.topic || !postData.content) {
        console.warn("Received invalid post data:", postData);
        return;
    }

    const postElement = document.createElement('article');
    postElement.classList.add('post');

    const titleElement = document.createElement('h2');
    titleElement.textContent = postData.topic;

    const contentElement = document.createElement('div');
    contentElement.classList.add('post-content'); // Add class for potential specific styling

    // Use marked.parse() to convert Markdown to HTML
    // Ensure marked.js is loaded (added in index.html)
    if (typeof marked === 'function') {
        // Basic sanitization is often recommended with markdown parsers
        // if the source isn't fully trusted, but here we trust our backend.
        // marked.js has options for sanitization if needed.
        contentElement.innerHTML = marked.parse(postData.content);
    } else {
        console.error("marked.js library not loaded!");
        // Fallback to plain text rendering if marked.js fails
        const pre = document.createElement('pre');
        pre.textContent = postData.content;
        contentElement.appendChild(pre);
    }

    postElement.appendChild(titleElement);
    postElement.appendChild(contentElement);

    postsContainer.appendChild(postElement);
}

// --- displayError function remains the same ---
function displayError(message) {
     const errorElement = document.createElement('div');
     errorElement.classList.add('post', 'error-message'); // Add specific error class
     // errorElement.style.backgroundColor = '#e74c3c'; // Can use CSS class instead
     // errorElement.style.color = 'white';
     errorElement.innerHTML = `<h2>Error</h2><p>Could not load post: ${message}</p>`;
     postsContainer.appendChild(errorElement);
}

// --- checkScroll function remains the same ---
function checkScroll() {
    // Consider triggering slightly earlier if posts are very long
    const triggerHeight = document.documentElement.scrollHeight - window.innerHeight * 2.0; // Trigger earlier
    if (window.scrollY > triggerHeight && !isLoading) {
        console.log("Scroll threshold reached, fetching new post...");
        fetchPost();
    }
}

// --- initialLoad function remains the same (but loads fewer posts initially) ---
async function initialLoad() {
    console.log(`Loading initial ${initialLoadCount} post(s)...`);
    for (let i = 0; i < initialLoadCount; i++) {
        await fetchPost();
         // No delay needed if loading just one
    }
    console.log("Initial load complete.");
}

// Event Listeners remain the same
window.addEventListener('scroll', checkScroll);
document.addEventListener('DOMContentLoaded', initialLoad);