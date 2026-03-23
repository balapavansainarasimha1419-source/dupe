import os
import sys
import logging
import chromadb
import difflib
import re
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
    Handles Vector Storage (ChromaDB) and AI Feature Engineering (SentenceTransformers + HDBSCAN)
    for the FileSense app. Runs 100% offline and is strictly air-gapped.
    """
    
    def __init__(self):
        try:
            self.model = SentenceTransformer(config.MODEL_NAME)  # FIX 1: use config constant, not hardcoded string
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
    
    def clear_database(self) -> bool:
        """
        Completely wipes the AI's memory by deleting and recreating the collection.
        This is much safer and faster than deleting files one by one.
        """
        if not self.chroma_client:
            return False
        try:
            # Annihilate the entire collection
            self.chroma_client.delete_collection("filesense_docs")
            # Create a fresh, empty one in its place
            self.collection = self.chroma_client.get_or_create_collection(name="filesense_docs")
            return True
        except Exception as e:
            import logging
            logging.error(f"Error clearing database: {e}")
            return False
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

    # FIX 2: DELETED the first get_file_metadata() that lived here (old lines 61-73).
    # It used ChromaDB 'ids' as dict keys which was WRONG for delta sync.
    # The correct version below uses 'filepath' from metadata as the key.

    def remove_file(self, filepath: str) -> bool:
        """Removes a specific deleted file from the AI's memory."""
        if not self.collection:
            return False
        try:
            # FIX 3: ID in ChromaDB is now the filepath itself (see add_file fix below),
            # so we delete by filepath directly. This replaces the old uuid-based delete.
            self.collection.delete(ids=[filepath])
            return True
        except Exception as e:
            logging.error(f"Error deleting file {filepath}: {e}")
            return False

    def add_file(self, filename: str, filepath: str, text: str, mtime: float = 0.0):
        """
        Adds or UPDATES a document and its modification timestamp in the database.
        Uses upsert so re-scanning a modified file replaces the old record cleanly.
        """
        if not self.collection or not self.model or not text.strip():
            return False
            
        try:
            # FIX 4: Use filepath as the stable document ID instead of uuid4().
            # This is the core fix — it means the same file can never be duplicated.
            # upsert() below will UPDATE the existing record if filepath already exists.
            doc_id = filepath

            # Convert the text into an AI embedding
            embedding = self._generate_embedding(text)
            
            if embedding:
                # FIX 5: collection.upsert() instead of collection.add().
                # add() crashes if the ID already exists. upsert() safely replaces it.
                self.collection.upsert(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": filename, 
                        "filepath": filepath,
                        "mtime": mtime
                    }],
                    ids=[doc_id]
                )
                return True
        except Exception as e:
            logging.error(f"Error adding file to DB: {e}")
            return False

    def get_file_metadata(self) -> dict:
        """
        Retrieves all indexed files and their last modified timestamps.
        Returns a dictionary formatted as {filepath: mtime} for Delta Syncing.
        """
        if not self.collection:
            return {}
            
        try:
            # We only need the metadata, no need to load heavy documents into RAM
            results = self.collection.get(include=['metadatas'])
            metas = results.get('metadatas', [])
            
            file_dict = {}
            for meta in metas:
                if meta:
                    # Safely grab the filepath
                    path = meta.get('filepath') or meta.get('path')
                    if path:
                        # Grab the timestamp. If it's an older file without one, default to 0.0
                        file_dict[path] = meta.get('mtime', 0.0)
                        
            return file_dict
        except Exception as e:
            logging.error(f"Error retrieving metadata: {e}")
            return {}

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
                n_jobs=-1  # Uses all available CPU cores for speed
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
        Hybrid Search: Combines AI Semantic Search with Case-Insensitive Keyword Matching
        and a Fuzzy fallback for misspellings.
        """
        # FIX 6: Guard against model or collection being None.
        # Previously self.model.encode() would crash with AttributeError if model failed to load.
        if not self.model or not self.collection:
            return {"error": "AI engine not ready. Check startup logs for errors."}

        if not query_text.strip():
            return {"error": "Empty search query."}

        try:
            matches = {}
            query_lower = query_text.lower()
            query_words = query_lower.split()

            # ==========================================
            # 1. LEXICAL SEARCH (Case-Insensitive Substring Match)
            # ==========================================
            all_records = self.collection.get(include=['metadatas', 'documents'])
            
            if all_records and all_records.get('documents'):
                docs = all_records['documents']
                metas = all_records['metadatas']
                
                for doc, meta in zip(docs, metas):
                    filepath = meta.get('filepath', 'Unknown')
                    doc_lower = doc.lower()
                    filename_lower = meta.get('filename', '').lower()

                    is_match = False

                    # Direct substring match in content or filename
                    if query_lower in doc_lower or query_lower in filename_lower:
                        is_match = True

                    # FIX 7: Fuzzy fallback for misspellings — raised cutoff from 0.50 to 0.75
                    # to prevent false positives. Only fires if direct match fails.
                    # This replaces the dead search() method's fuzzy logic and makes it
                    # actually reachable from the UI.
                    if not is_match:
                        filename_words = set(re.findall(r'\b\w+\b', filename_lower))
                        doc_vocabulary = set(re.findall(r'\b\w+\b', doc_lower))
                        total_vocabulary = filename_words.union(doc_vocabulary)

                        for q_word in query_words:
                            # 0.75 cutoff: "resume" matches "resumé", "python" matches "pyhton"
                            # but NOT "cat" matching "car" or other false positives
                            if difflib.get_close_matches(q_word, total_vocabulary, n=1, cutoff=0.75):
                                is_match = True
                                break

                    if is_match:
                        matches[filepath] = {
                            "filename": meta.get('filename', 'Unknown'),
                            "filepath": filepath,
                            "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                            "distance": "Keyword Match",
                            "score": 0.0
                        }

            # ==========================================
            # 2. SEMANTIC SEARCH (AI Vector Match)
            # ==========================================
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
                return {"error": "No matches found (neither keyword nor semantic)."}

            final_results = list(matches.values())
            final_results.sort(key=lambda x: x['score'])

            return {"matches": final_results[:top_k]}

        except Exception as e:
            return {"error": str(e)}
