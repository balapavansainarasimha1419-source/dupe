import os
import sys
import logging
import chromadb
from sklearn.cluster import HDBSCAN

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
    # PART C: Clustering Module (HDBSCAN Logic)
    # ==========================================
    def cluster_files(self, min_cluster_size: int = 2) -> dict:
        """
        Organizes files into semantic clusters using HDBSCAN.
        Automatically determines the number of clusters and isolates noise/outliers.
        """
        if not self.collection:
            return {'error': 'Database unavailable'}
            
        try:
            data = self.collection.get(include=['embeddings', 'metadatas'])
            embeddings = data.get('embeddings')
            metadatas = data.get('metadatas')
            
            # Check if we have enough files to form even a single minimum-sized cluster
            if embeddings is None or len(embeddings) < min_cluster_size:
                total_files = len(embeddings) if embeddings is not None else 0
                return {
                    'warning': f'Not enough files ({total_files}) to form a cluster. '
                               f'Please scan at least {min_cluster_size} files!'
                }
                
            # Initialize HDBSCAN
            # metric='euclidean' works well with SentenceTransformers default embeddings
            hdb = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                metric='euclidean',
                n_jobs=-1 # Uses all available CPU cores for speed
            )
            
            labels = hdb.fit_predict(embeddings)
            
            clusters = {}
            for label, meta in zip(labels, metadatas):
                filename = meta.get('filename', 'Unknown')
                
                # HDBSCAN assigns the label -1 to data points it considers "noise"
                if label == -1:
                    cluster_name = "Uncategorized / Noise"
                else:
                    cluster_name = f"Cluster {label}"
                    
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                    
                clusters[cluster_name].append(filename)
                
            return clusters
            
        except Exception as e:
            logging.error(f"Error during HDBSCAN clustering: {e}")
            return {'error': f"Clustering failed: {str(e)}"}
    

    def search_documents(self, query_text: str, top_k: int = 5, distance_threshold: float = 1.5):
        """
        Hybrid Search: Combines AI Semantic Search with Case-Insensitive Keyword Matching.
        """
        if not query_text.strip():
            return {"error": "Empty search query."}

        try:
            matches = {}
            # Convert the user's query to lowercase right away
            query_lower = query_text.lower()

            # ==========================================
            # 1. LEXICAL SEARCH (Case-Insensitive Match)
            # ==========================================
            # Fetch records and use Python's .lower() to bypass ChromaDB's case-sensitivity
            all_records = self.collection.get(include=['metadatas', 'documents'])
            
            if all_records and all_records.get('documents'):
                docs = all_records['documents']
                metas = all_records['metadatas']
                
                for doc, meta in zip(docs, metas):
                    # Check if the lowercase query exists in the lowercase document
                    if query_lower in doc.lower():
                        filepath = meta.get('filepath', 'Unknown')
                        matches[filepath] = {
                            "filename": meta.get('filename', 'Unknown'),
                            "filepath": filepath,
                            "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                            "distance": "Exact Match", 
                            "score": 0.0 
                        }

            # ==========================================
            # 2. SEMANTIC SEARCH (AI Vector Match)
            # ==========================================
            # Embeddings naturally handle case-insensitivity well
            query_embedding = self.model.encode(query_text).tolist()
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )

            v_docs = vector_results.get('documents', [[]])[0]
            v_metas = vector_results.get('metadatas', [[]])[0]
            v_dists = vector_results.get('distances', [[]])[0]

            for doc, meta, dist in zip(v_docs, v_metas, v_dists):
                filepath = meta.get('filepath', 'Unknown')
                
                if dist <= distance_threshold and filepath not in matches:
                    matches[filepath] = {
                        "filename": meta.get('filename', 'Unknown'),
                        "filepath": filepath,
                        "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                        "distance": round(dist, 4),
                        "score": dist
                    }

            # ==========================================
            # 3. MERGE & SORT RESULTS
            # ==========================================
            if not matches:
                return {"error": "No matches found (neither exact keyword nor semantic)."}

            final_results = list(matches.values())
            final_results.sort(key=lambda x: x['score'])

            return {"matches": final_results[:top_k]}

        except Exception as e:
            return {"error": str(e)}
