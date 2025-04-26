// Import the required Firebase modules
import { initializeApp } from 'firebase/app';
import { 
  getFirestore, 
  collection, 
  doc,
  getDoc,
  getDocs,
  addDoc,
  updateDoc,
  deleteDoc,
  query,
  where 
} from 'firebase/firestore';

// Your Firebase configuration
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firestore
const db = getFirestore(app);

// Example functions for Firestore operations

// Create a new document
async function addDocument(collectionName, data) {
  try {
    const docRef = await addDoc(collection(db, collectionName), data);
    console.log("Document written with ID: ", docRef.id);
    return docRef.id;
  } catch (error) {
    console.error("Error adding document: ", error);
    throw error;
  }
}

// Read a single document by ID
async function getDocument(collectionName, documentId) {
  try {
    const docRef = doc(db, collectionName, documentId);
    const docSnap = await getDoc(docRef);
    
    if (docSnap.exists()) {
      return docSnap.data();
    } else {
      console.log("No such document!");
      return null;
    }
  } catch (error) {
    console.error("Error getting document: ", error);
    throw error;
  }
}

// Read all documents in a collection
async function getAllDocuments(collectionName) {
  try {
    const querySnapshot = await getDocs(collection(db, collectionName));
    const documents = [];
    querySnapshot.forEach((doc) => {
      documents.push({ id: doc.id, ...doc.data() });
    });
    return documents;
  } catch (error) {
    console.error("Error getting documents: ", error);
    throw error;
  }
}

// Update a document
async function updateDocument(collectionName, documentId, data) {
  try {
    const docRef = doc(db, collectionName, documentId);
    await updateDoc(docRef, data);
    console.log("Document successfully updated!");
  } catch (error) {
    console.error("Error updating document: ", error);
    throw error;
  }
}

// Delete a document
async function deleteDocument(collectionName, documentId) {
  try {
    await deleteDoc(doc(db, collectionName, documentId));
    console.log("Document successfully deleted!");
  } catch (error) {
    console.error("Error deleting document: ", error);
    throw error;
  }
}

// Query documents with conditions
async function queryDocuments(collectionName, fieldName, operator, value) {
  try {
    const q = query(collection(db, collectionName), where(fieldName, operator, value));
    const querySnapshot = await getDocs(q);
    const documents = [];
    querySnapshot.forEach((doc) => {
      documents.push({ id: doc.id, ...doc.data() });
    });
    return documents;
  } catch (error) {
    console.error("Error querying documents: ", error);
    throw error;
  }
}

// Example usage:
/*
// Add a new document
const newUser = {
  name: "John Doe",
  email: "john@example.com",
  age: 30
};
const newDocId = await addDocument('users', newUser);

// Get a document
const user = await getDocument('users', newDocId);

// Get all documents
const allUsers = await getAllDocuments('users');

// Update a document
await updateDocument('users', newDocId, { age: 31 });

// Query documents
const youngUsers = await queryDocuments('users', 'age', '<', 25);

// Delete a document
await deleteDocument('users', newDocId);
*/
