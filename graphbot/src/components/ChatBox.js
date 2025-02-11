import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatBox.css';

const ChatBox = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleClearChat = () => {
        setMessages([]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { text: userMessage, type: 'user' }]);
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:5001/generate_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                data.messages.forEach(part => {
                    if (part.type === 'text') {
                        setMessages(prev => [...prev, { text: part.content, type: 'bot' }]);
                    } else if (part.type === 'graph') {
                        setMessages(prev => [...prev, { type: 'bot-image', imageId: part.content }]);
                    }
                });
            } else {
                setMessages(prev => [...prev, {
                    text: data.error || 'An error occurred while generating the response.',
                    type: 'error'
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                text: 'Failed to communicate with the server. Please try again.',
                type: 'error'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const renderMessage = (message, index) => {
        const messageClass = `message ${message.type}`;

        if (message.type === 'bot-image') {
            return (
                <div key={index} className={messageClass}>
                    <div className="message-content">
                        <img
                            src={`http://localhost:5001/generated_graphs/${message.imageId}.png`}
                            alt="Generated Graph"
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.style.display = 'none';
                                e.target.parentElement.innerHTML = 'Failed to load graph image.';
                            }}
                        />
                    </div>
                </div>
            );
        }

        return (
            <div key={index} className={messageClass}>
                <div className="message-content">
                    {message.type === 'bot' ? (
                        <ReactMarkdown>{message.text}</ReactMarkdown>
                    ) : (
                        message.text
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="chat-container">
            <div className="chat-header">
                <h2>GraphBot Chat</h2>
                <button
                    onClick={handleClearChat}
                    className="clear-button"
                    title="Clear chat"
                >
                    Clear Chat
                </button>
            </div>
            <div className="messages-container">
                {messages.length === 0 && (
                    <div className="welcome-message">
                        Welcome to GraphBot! Ask me to create any graph or visualization for you.
                    </div>
                )}
                {messages.map((message, index) => renderMessage(message, index))}
                {isLoading && (
                    <div className="message bot">
                        <div className="message-content loading">Thinking</div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} className="input-form">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading || !input.trim()}>
                    Send
                </button>
            </form>
        </div>
    );
};

export default ChatBox;
