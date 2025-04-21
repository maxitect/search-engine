# inference.py
import faiss
import numpy as np

class SearchEngine:
    def __init__(self, model, documents):
        self.model = model
        self.documents = documents
        self.index = None
        self.doc_embeddings = []
        
    def build_index(self, batch_size=32):
        """Pre-compute document embeddings"""
        doc_loader = DataLoader(self.documents, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in doc_loader:
                batch = batch.cuda()
                emb = model.encode_doc(batch).cpu().numpy()
                self.doc_embeddings.append(emb)
        
        self.doc_embeddings = np.concatenate(self.doc_embeddings)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)
    
    def search(self, query, top_k=5):
        """Search for relevant documents"""
        with torch.no_grad():
            query_emb = model.encode_query(query).cpu().numpy()
        
        distances, indices = self.index.search(query_emb, top_k)
        return [(self.documents[i], distances[0][j]) 
                for j, i in enumerate(indices[0])]