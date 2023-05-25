import "./normal.css";
import "./App.css";
import retrieveInformation from "./callAPI";
import { useState, useEffect } from "react";
import SideMenu from "./SideMenu";
import ChatBox from "./ChatBox";

function App() {
  // This is a very stupid workaround for useEffect complaining about not being used. Program doesn't run if I remove it though.
  //useEffect(() => {}, []);
  const [chatInput, setChatInput] = useState("");
  const [chatLog, setChatLog] = useState([
    {
      user: "gpt",
      message: "How can I help you today?",
    },
  ]);

  // clear chats
  function clearChat() {
    setChatLog([]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    let chatLogNew = [...chatLog, { user: "me", message: `${chatInput}` }];
    setChatInput("");
    setChatLog(chatLogNew);
    
    // fetch response to the api combining the chat log array of messages and seinding it as a message to localhost:3000 as a post
    const messages = chatLogNew.map((message) => message.message).join("\n");

    // Call API in retreiveInfo...
    const data = await retrieveInformation(messages);

    setChatLog([...chatLogNew, { user: "gpt", message: `${data}` }]);
  }

  // Scroll
  useEffect(() => {
    const scrollToTheBottomChatLog = document.getElementsByClassName("chat-log")[0];
    scrollToTheBottomChatLog.scrollTop = scrollToTheBottomChatLog.scrollHeight;
  }, [chatLog]);

  return (
    <div className="App">
      <SideMenu
        clearChat={clearChat}
      />
      <ChatBox
        chatInput={chatInput}
        chatLog={chatLog}
        setChatInput={setChatInput}
        handleSubmit={handleSubmit}
      />
    </div>
  );
}

export default App;
