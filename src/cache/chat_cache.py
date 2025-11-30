from typing import Optional
from src.agent.chat_agent import ChatAgent


class ChatCache:
    """In-memory storage for managing ChatAgent instances."""

    def __init__(self):
        """Initialize the in-memory cache."""
        self._agents: dict[str, ChatAgent] = {}

    def store(self, chat_id: str, chat_agent: ChatAgent) -> bool:
        """
        Store a ChatAgent instance in memory.

        Args:
            chat_id: The unique identifier for the chat session
            chat_agent: The ChatAgent instance to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._agents[chat_id] = chat_agent
            return True
        except Exception as e:
            print(f"Error storing chat agent {chat_id}: {e}")
            return False

    def get(self, chat_id: str) -> Optional[ChatAgent]:
        """
        Retrieve a ChatAgent instance from memory.

        Args:
            chat_id: The unique identifier for the chat session

        Returns:
            ChatAgent instance if found, None otherwise
        """
        return self._agents.get(chat_id)

    def exists(self, chat_id: str) -> bool:
        """
        Check if a chat agent exists in memory.

        Args:
            chat_id: The unique identifier for the chat session

        Returns:
            bool: True if exists, False otherwise
        """
        return chat_id in self._agents

    def delete(self, chat_id: str) -> bool:
        """
        Delete a ChatAgent instance from memory.

        Args:
            chat_id: The unique identifier for the chat session

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if chat_id in self._agents:
                del self._agents[chat_id]
                return True
            return False
        except Exception as e:
            print(f"Error deleting chat agent {chat_id}: {e}")
            return False

    def get_all_keys(self) -> list[str]:
        """
        Get all chat agent keys from memory.

        Returns:
            List of chat IDs
        """
        return list(self._agents.keys())


# Singleton instance
chat_cache = ChatCache()
