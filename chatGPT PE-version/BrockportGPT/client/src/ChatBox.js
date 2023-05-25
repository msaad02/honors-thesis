import OpenAISVGLogo from './OpenAISVGLogo'
import UserLogo from './UserLogo'
// import DOMPurify from 'dompurify';

// Primary Chat Window
const ChatBox = ({chatLog, setChatInput, handleSubmit, chatInput}) =>
  <section className="chatbox">
      <div className="chat-log">
        {chatLog.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
      </div>
        <div className="chat-input-holder">
      <form className="form" onSubmit={handleSubmit}>
          <input 
          rows="1"
          value={chatInput}
          onChange={(e)=> setChatInput(e.target.value)}
          placeholder={"Type your message to BrockportGPT here!"}
          className="chat-input-textarea" ></input>
          <button className="submit" type="submit">Submit</button>
          </form>
        </div>
      </section>

// Individual Chat Message

const ChatMessage = ({ message }) => {
  return (
    <div className={`chat-message ${message.user === "gpt" && "chatgpt"}`}>
    <div className="chat-message-center">
      <div className={`ava area ${message.user === "gpt" && "blahblah"}`}>
        {message.user === "gpt" ? <OpenAISVGLogo /> : <UserLogo />}
      </div>
      <div className="message">
        {message.message}
        
      </div>
    </div>
  </div>
  )
}

// const ChatMessage = ({ message }) => {
//   const processedMessage = processMessage(message.message);
//   const sanitizedMessage = DOMPurify.sanitize(processedMessage);

//   return (
//     <div className={`chat-message ${message.user === "gpt" && "chatgpt"}`}>
//       <div className="chat-message-center">
//         <div className={`ava area ${message.user === "gpt" && "blahblah"}`}>
//           {message.user === "gpt" ? <OpenAISVGLogo /> : <UserLogo />}
//         </div>
//         <div className="message">
//           <div dangerouslySetInnerHTML={{ __html: sanitizedMessage }} />
//         </div>
//       </div>
//     </div>
//   )
// }

// function processMessage(message) {
//   // Regular expression to identify numbered lists
//   const regex = /((?:\d+\.)(?:\s*\S.*)(?:[\n\r]+(?=\d+\.))*)/gs;

//   // Wrap the matched lists with <ol> and <li> tags
//   const processedMessage = message.replace(regex, (match) => {
//     const items = match.trim().split(/\n|\r/).map((item) => item.replace(/^\d+\.\s*/, '').trim());
//     const listItems = items.map((item, index) => `<li key="${index}">${item}</li>`).join('');
//     return `<ol class="numbered-list">${listItems}</ol>`;
//   });

//   return processedMessage;
// }





export default ChatBox