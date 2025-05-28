import streamlit as st
import os
from ticket_matching_system import TicketMatchingSystem
from ticket_resolution_system import TicketResolutionSystem

# Paths to pre-built index and data
INDEX_PATH = "ticket_index.bin"  # Ensure this file exists
BASE_DF_PATH = "../data/combined_data.csv"  # Ensure this file exists

# Load the Ticket Matching System with pre-built index
matching_system = TicketMatchingSystem(index_path=INDEX_PATH, resolved_tickets_data_path=BASE_DF_PATH)

# Load the Ticket Resolution System
resolution_system = TicketResolutionSystem()

# Streamlit UI
st.title("Helpdesk Ticket Resolution Assistant 🛠️")

# User input fields
issue = st.text_input("Issue", placeholder="Enter the issue (e.g., Printer not working)")
category = st.text_input("Category", placeholder="Enter the category (e.g., Hardware, Software)")
description = st.text_area("Description", placeholder="Enter detailed issue description")

if st.button("Find Resolution"):
    if issue and category and description:
        # Create a new ticket
        new_ticket = {"Issue": issue, "Category": category, "Description": description}
        
        # Find similar tickets
        with st.spinner("🔍 Searching for similar tickets..."):
            similar_tickets = matching_system.find_similar_tickets(issue, category, description, k=3)
        
        # Display similar tickets
        st.subheader("Similar Tickets From Past :")
        if similar_tickets:
            for ticket in similar_tickets:
                st.write(f"**Issue:** {ticket['issue']}")
                st.write(f"**Category:** {ticket['category']}")
                st.write(f"**Description:** {ticket['description']}")
                st.write(f"**Resolved:** {'✅ Yes' if ticket['resolved'] else '❌ No'}")
                if ticket['resolved']:
                    st.write(f"**Resolution:** {ticket['resolution']}")
                st.write("---")
        else:
            st.write("No similar tickets found.")

        # Generate AI response
        with st.spinner("🧠 Generating AI response..."):
            response = resolution_system.generate_response(new_ticket, similar_tickets)
        
        # Display AI response
        st.subheader("AI-Generated Resolution:")
        st.write(response)
    else:
        st.warning("Please fill in all fields.")
