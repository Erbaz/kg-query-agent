import asyncio
import uuid
from src.agent.chat_agent import ChatAgent


async def main():
    # Create a test UUID
    test_uuid = str(uuid.uuid4())
    
    # Instantiate ChatAgent
    print(f"Creating ChatAgent with UUID: {test_uuid}")
    print("=" * 60)
    chat_agent = ChatAgent(uuid=test_uuid)
    
    print("\nInteractive Chat Shell - Testing Context Handling")
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print("=" * 60)
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nEnding conversation. Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Call chat_stream and display response
        print("\nAgent: ", end="", flush=True)
        try:
            response = await chat_agent.chat_stream(user_input)
            print()  # New line after streaming response
        except Exception as e:
            print(f"\nError: {e}")
            break


if __name__ == "__main__":
    asyncio.run(main())

