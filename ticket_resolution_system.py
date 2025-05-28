import json
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient
import os

from ticket_matching_system import TicketMatchingSystem

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

class TicketResolutionSystem:
    def __init__(self, model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initialize the Ticket Resolution System.
        
        Args:
            model_id: The HuggingFace model ID to use for inference
        """
        self.client = InferenceClient(model=model_id, token=HF_API_KEY)
        
    def generate_response(self, new_ticket: Dict[str, Any], similar_tickets: List[Dict[str, Any]]) -> str:
        """
        Generate a coherent response for a new ticket based on similar past tickets.
        
        Args:
            new_ticket: Dictionary containing the new ticket information
            similar_tickets: List of dictionaries containing similar tickets from vector search
        
        Returns:
            A coherent response that can be used by a human agent
        """
        # Check if any tickets were found
        if not similar_tickets:
            # Generate a simple suggestion using the model with high confidence settings
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a technical support assistant. Provide a very brief solution suggestion (max 15 words) for the following issue ONLY if you are highly confident. If not confident, respond with 'No immediate solution available.'"},
                    {"role": "user", "content": f"Issue: {json.dumps(new_ticket, indent=2)}"}
                ],
                max_tokens=50,
                temperature=0.1,  # Very low temperature for high confidence
                top_p=0.1
            )
            suggestion = response['choices'][0]['message']['content']
            return f"No matching tickets found in the database.\n\nSuggested direction: {suggestion}\n\nBest, your Smart assistant"

        # Convert similar_tickets to a JSON-serializable format
        serializable_tickets = []
        for ticket in similar_tickets:
            serializable_ticket = {
                "ticket_id": str(ticket["ticket_id"]),
                "similarity_score": float(ticket["similarity_score"]),
                "issue": str(ticket["issue"]),
                "category": str(ticket["category"]),
                "description": str(ticket["description"]),
                "resolved": bool(ticket["resolved"]),
                "resolution": str(ticket["resolution"]) if ticket["resolution"] else ""
            }
            serializable_tickets.append(serializable_ticket)
        
        # Check if any of the similar tickets are resolved
        resolved_tickets = [ticket for ticket in serializable_tickets if ticket.get("resolved")]
        
        # Craft the prompt based on the situation
        if resolved_tickets:
            prompt = self._craft_prompt_for_resolved_tickets(new_ticket, resolved_tickets)
        else:
            prompt = self._craft_prompt_for_unresolved_tickets(new_ticket, serializable_tickets)
        
        # Generate response using the HuggingFace model
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"New Ticket: {json.dumps(new_ticket, indent=2)}"}
            ],
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.95
        )
        
        # Extract the generated response
        generated_response = response['choices'][0]['message']['content']
        return generated_response
    
    def _craft_prompt_for_resolved_tickets(self, new_ticket: Dict[str, Any], resolved_tickets: List[Dict[str, Any]]) -> str:
        """
        Craft a prompt for when there are resolved similar tickets.
        """
        prompt = f"""
        You are an AI assistant that helps Human Agents respond to support tickets. 
        
        I will provide you with a new support ticket and details from {len(resolved_tickets)} similar resolved tickets from our database.
        
        Your task is to:
        1. Analyze the new ticket and the resolved similar tickets
        2. Create a coherent response that addresses the new ticket's issue
        3. Include the most relevant solution from the resolved tickets
        4. End the message by saying: Best, your Smart assistant 
        Here are the similar resolved tickets:
        {json.dumps(resolved_tickets, indent=2)}
        
        Please create a response that the agent can use to address the new ticket. Be concise but comprehensive.
        """
        return prompt
    
    def _craft_prompt_for_unresolved_tickets(self, new_ticket: Dict[str, Any], similar_tickets: List[Dict[str, Any]]) -> str:
        """
        Craft a prompt for when there are no resolved similar tickets.
        """
        prompt = f"""
        You are an AI assistant that helps Human Agents respond to support tickets.
        
        I will provide you with a new support ticket and details from {len(similar_tickets)} similar tickets from our database, but none of these similar tickets have been resolved.
        
        Your task is to:
        1. Analyze the new ticket and the similar unresolved tickets
        2. Create a coherent response that acknowledges the ongoing nature of this issue
        3. Share details about the similar tickets and what approaches did not work
        4. Suggest potential next steps based on the history of attempts
        5. Format your response to be ready for a human agent to review and send
        6. End the message by saying: Best, your Smart assistant
        
        Here are the similar unresolved tickets:
        {json.dumps(similar_tickets, indent=2)}
        
        Please create a response that the agent can use to address the new ticket, acknowledging that we don't have a proven solution yet.
        """
        return prompt

# Example usage
if __name__ == "__main__":
    # Initialize the TicketMatchingSystem
    system = TicketMatchingSystem(index_path="ticket_index.bin", resolved_tickets_data_path="data/combined_data.csv")
    
    # Initialize the TicketResolutionSystem
    resolution_system = TicketResolutionSystem()
    
    # Sample new ticket
    # new_ticket = {
    #     "Issue": "Cannot connect to VPN after system update",
    #     "Category": "Network",
    #     "Description": "After updating my Windows to the latest version, I'm unable to connect to the company VPN. I get an error saying 'Connection failed: timeout'."
    # }


    # new_ticket = {
    #     'Issue': 'Printer not connecting to WiFi',
    #     'Category': 'Hardware',
    #     'Description': 'WiFi printer is not connecting to any devices in the office.'
    # }
    new_ticket = {
        'Issue': 'Virus discovered',
        'Category': 'Malware',
        'Description': 'There is a malware discovered circulating in office machines. We need immediate cleanup!'
    }
    
    # Find similar tickets using the TicketMatchingSystem
    similar_tickets = system.find_similar_tickets(
        new_ticket["Issue"],
        new_ticket["Category"],
        new_ticket["Description"],
        k=3
    )
    
    print("User Query: ", new_ticket)
    print("\nMatching tickets:")
    for i, result in enumerate(similar_tickets, 1):
        print(f"\n{i}. Ticket ID: {result['ticket_id']}")
        print(f"   Issue: {result['issue']}")
        print(f"   Category: {result['category']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Resolved: {result['resolved']}")
        if result['resolved']:
            print(f"   Resolution: {result['resolution']}")
        print(f"   Description: {result['description']}")
        
    # Generate the response using the TicketResolutionSystem
    response = resolution_system.generate_response(new_ticket, similar_tickets)
    print(response)