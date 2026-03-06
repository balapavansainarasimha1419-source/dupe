import os
import sys
import logging
import chromadb
from sklearn.cluster import KMeans

# Bypass the strict Intel OpenMP DLL conflict (very common with PyTorch on Windows)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Force Windows to look inside Anaconda's Library/bin folder for missing C++ DLLs
if sys.platform == 'win32':
    env_path = os.path.dirname(sys.executable)
    bin_path = os.path.join(env_path, 'Library', 'bin')
    if os.path.exists(bin_path):
        os.add_dll_directory(bin_path)

from sentence_transformers import SentenceTransformer
import config

class VectorDB:
    """
    Handles Vector Storage (ChromaDB) and AI Feature Engineering (SentenceTransformers + KMeans)
    for the FileSense app. Runs 100% offline and is strictly air-gapped.
    """
    
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.error(f"Critical error loading SentenceTransformer: {e}")
            self.model = None

        try:
            self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
            self.collection = self.chroma_client.get_or_create_collection(name="filesense_docs")
        except Exception as e:
            logging.error(f"Critical error initializing ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

    # ==========================================
    # PART A: Feature Engineering Module
    # ==========================================
    def _generate_embedding(self, text: str) -> list[float]:
        """Cleans text and generates a local vector embedding."""
        if not self.model or not text:
            return []
            
        try:
            cleaned_text = text.replace('\n', ' ').strip()
            return self.model.encode(cleaned_text).tolist()
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return []

    # ==========================================
    # PART B: Memory Management & Search (ChromaDB)
    # ==========================================
    def get_file_metadata(self) -> dict:
        """Returns a dict mapping filepaths to their last modified timestamp."""
        if not self.collection:
            return {}
        try:
            data = self.collection.get(include=['metadatas'])
            ids = data.get('ids', [])
            metadatas = data.get('metadatas', [])
            # Return {filepath: timestamp} so we know exactly when the AI last read it
            return {doc_id: meta.get('last_modified', 0.0) for doc_id, meta in zip(ids, metadatas)}
        except Exception as e:
            logging.error(f"Error fetching metadata: {e}")
            return {}

    def remove_file(self, filepath: str) -> bool:
        """Removes a specific deleted file from the AI's memory."""
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[filepath])
            return True
        except Exception as e:
            logging.error(f"Error deleting file {filepath}: {e}")
            return False

    def add_file(self, filename: str, filepath: str, text: str) -> bool:
        """Generates an embedding and safely stores/updates the file data in ChromaDB."""
        if not self.collection:
            return False
            
        try:
            embedding = self._generate_embedding(text)
            if not embedding:
                return False
            
            # Grab the exact time the file was last edited on the computer
            last_modified = os.path.getmtime(filepath)
                
            # UPSERT: If the file is new, it adds it. If it already exists, it overwrites it!
            self.collection.upsert(
                ids=[filepath],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{'filename': filename, 'path': filepath, 'last_modified': last_modified}]
            )
            return True
        except Exception as e:
            logging.error(f"Error adding file {filename} to ChromaDB: {e}")
            return False

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Queries ChromaDB and filters out highly irrelevant results using a distance threshold."""
        if not self.collection or not self.model:
            return []
            
        try:
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            clean_results = []
            if results and results.get('metadatas') and results['metadatas'][0]:
                for i in range(len(results['metadatas'][0])):
                    distance = results['distances'][0][i]
                    
                    if distance < 1.5:
                        meta = results['metadatas'][0][i]
                        doc = results['documents'][0][i]
                        
                        clean_results.append({
                            'filename': meta.get('filename', 'Unknown'),
                            'filepath': meta.get('path', 'Unknown'),
                            'text_content': doc
                        })
            return clean_results
            
        except Exception as e:
            logging.error(f"Error executing search query: {e}")
            return []

    # ==========================================
    # PART C: Clustering Module (K-Means Logic)
    # ==========================================
    def cluster_files(self, n_clusters: int = 3) -> dict:
        """
        Organizes files into semantic clusters, with fixes for Truth Value ambiguity.
        """
        if not self.collection:
            return {'error': 'Database unavailable'}
            
        try:
            data = self.collection.get(include=['embeddings', 'metadatas'])
            embeddings = data.get('embeddings')
            metadatas = data.get('metadatas')
            
            if embeddings is None or len(embeddings) < n_clusters:
                total_files = len(embeddings) if embeddings is not None else 0
                return {'warning': f'Not enough files ({total_files}) to form {n_clusters} clusters. Please scan at least {n_clusters} files!'}
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            
            clusters = {i: [] for i in range(n_clusters)}
            for label, meta in zip(labels, metadatas):
                filename = meta.get('filename', 'Unknown')
                clusters[label].append(filename)
                
            return clusters
            
        except Exception as e:
            logging.error(f"Error during K-Means clustering: {e}")
            return {'error': f"Clustering failed: {str(e)}"}