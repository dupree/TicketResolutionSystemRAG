import os
import hnswlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class TicketMatchingSystem:
    def __init__(self, resolved_tickets_data_path='data/combined_data.csv', model_name='sentence-transformers/all-MiniLM-L6-v2', index_path=None):
        """
        Initialize the TicketMatchingSystem.
        
        Args:
            resolved_tickets_data_path (str): Path to CSV file containing resolved ticket data. Required parameter.
            model_name (str): Name of the sentence transformer model.
            index_path (str, optional): Path to a pre-built index. If provided, the index will be loaded from disk.
        """
        if not resolved_tickets_data_path:
            print("Need a data file to initialize the system")
            return None

        self.model = SentenceTransformer(model_name)
        self.index = None
        self.ticket_ids = []
        self.resolved_tickets_data = None
        self.dim = 384  # default dimension for all-MiniLM-L6-v2
        self.index_path = index_path
        self.resolved_tickets_data_path = resolved_tickets_data_path

        # Check if data file exists
        if not os.path.exists(resolved_tickets_data_path):
            raise FileNotFoundError(f"Resolved tickets data file not found at {resolved_tickets_data_path}")

        # Load index if path provided, otherwise build new index from CSV
        if index_path:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found at {index_path}")
            self.load_index(index_path)
            self.load_resolved_tickets_data(resolved_tickets_data_path)
        else:
            # Build index from CSV
            self.build_index_from_csv(resolved_tickets_data_path)
    
    def create_ticket_string(self, issue, category, description):
        """Combine ticket fields into a single string representation"""
        if pd.isna(issue):
            issue = ""
        if pd.isna(category):
            category = ""
        if pd.isna(description):
            description = ""
        
        return f"{issue} {category} {description}".strip()
    
    def generate_embeddings(self, texts):
        """Generate embeddings for text(s)"""
        # Handle both single text and list of texts
        if isinstance(texts, str):
            return self.model.encode([texts])[0]
        else:
            return self.model.encode(texts)
    
    def build_index_from_csv(self, csv_path, save_path="ticket_index.bin"):
        """
        Build search index from CSV file of tickets and optionally save it to disk.
        
        Args:
            csv_path (str): Path to CSV file containing ticket data.
            save_path (str, optional): Path to save the index.
        """
        # Load CSV into DataFrame
        df = pd.read_csv(csv_path)
        self.resolved_tickets_data = df.copy()
        
        # Create ticket strings
        ticket_strings = []
        for _, row in df.iterrows():
            ticket_string = self.create_ticket_string(
                row.get('Issue', ''), 
                row.get('Category', ''), 
                row.get('Description', '')
            )
            ticket_strings.append(ticket_string)
        
        # Store ticket IDs
        self.ticket_ids = df['Ticket ID'].tolist()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(ticket_strings)
        
        # Build index
        n_elements = len(embeddings)
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=n_elements, ef_construction=200, M=16)
        self.index.add_items(embeddings, np.arange(n_elements))
        self.index.set_ef(50)  # ef influences search accuracy

        # Save index for re-use
        self.save_index(save_path)
    
    def save_index(self, save_path):
        """Save the index to disk"""
        if self.index is None:
            raise ValueError("Index has not been built yet")
        self.index.save_index(save_path)
        print(f"Index saved to {save_path}")
    
    def load_index(self, load_path):
        """Load the index from disk"""
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.load_index(load_path)
        self.index.set_ef(50)  # Set ef for search
        print(f"Index loaded from {load_path}")
    
    def load_resolved_tickets_data(self, resolved_tickets_data_path):
        """Load the resolved tickets DataFrame from disk"""
        self.resolved_tickets_data = pd.read_csv(resolved_tickets_data_path)
        self.ticket_ids = self.resolved_tickets_data['Ticket ID'].tolist()
        print(f"Resolved tickets data loaded from {resolved_tickets_data_path}")
    
    def find_similar_tickets(self, issue, category, description, k=3, similarity_threshold=0.5):
        """
        Find similar tickets to a query ticket
        
        Args:
            issue (str): Issue title
            category (str): Ticket category
            description (str): Ticket description
            k (int): Number of nearest neighbors to retrieve
            similarity_threshold (float): Minimum similarity score threshold (default: 0.5)
        """
        if self.index is None:
            raise ValueError("Index has not been built yet")
        if self.resolved_tickets_data is None:
            raise ValueError("Resolved tickets data has not been loaded yet")
        
        # Create ticket string and generate embedding
        query_string = self.create_ticket_string(issue, category, description)
        query_embedding = self.generate_embeddings(query_string)
        
        # Search for similar tickets
        labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=k)
        
        # Prepare results
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            similarity_score = 1 - dist
            
            # Only include results above similarity threshold
            if similarity_score >= similarity_threshold:
                ticket_id = self.ticket_ids[idx]
                
                # Get relevant info from original dataframe
                ticket_row = self.resolved_tickets_data[self.resolved_tickets_data['Ticket ID'] == ticket_id].iloc[0]
                
                results.append({
                    'ticket_id': ticket_id,
                    'similarity_score': similarity_score,
                    'issue': ticket_row.get('Issue', ''),
                    'category': ticket_row.get('Category', ''),
                    'description': ticket_row.get('Description', ''),
                    'resolved': ticket_row.get('Resolved', False),
                    'resolution': ticket_row.get('Resolution', ''),
                })
        
        # Sort by 'resolved' status (True first) and then by similarity score
        results.sort(key=lambda x: (-int(x['resolved']), -x['similarity_score']))
        
        return results

def test_system_with_existing_index():
    # Initialize system with an existing index and base DataFrame
    system = TicketMatchingSystem(
        index_path="ticket_index.bin",
        resolved_tickets_data_path="data/combined_data.csv"
    )

    # Test query
    # new_ticket = {
    #     'Issue': 'New software installation request',
    #     'Category': 'Software',
    #     'Description': 'A request to install new project management software.'
    # }
    # Test query
    new_ticket = {
        'Issue': 'Printer not connecting to WiFi',
        'Category': 'Hardware',
        'Description': 'WiFi printer is not connecting to any devices in the office.'
    }
    
    results = system.find_similar_tickets(
        new_ticket['Issue'],
        new_ticket['Category'],
        new_ticket['Description'],
        k=3
    )
    
    print("Query: ", new_ticket)
    print("\nMatching tickets:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Ticket ID: {result['ticket_id']}")
        print(f"   Issue: {result['issue']}")
        print(f"   Category: {result['category']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Resolved: {result['resolved']}")
        if result['resolved']:
            print(f"   Resolution: {result['resolution']}")
        print(f"   Description: {result['description']}")

if __name__ == "__main__":    
    # Test loading and using an existing index
    test_system_with_existing_index()