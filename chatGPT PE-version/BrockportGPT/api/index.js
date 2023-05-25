import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import morgan from "morgan";
import { operandClient, OperandService } from "@operandinc/sdk";
import { Configuration, OpenAIApi } from "openai";


// Open AI Configuration
const configuration = new Configuration({
  apiKey: "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",
});
const openai = new OpenAIApi(configuration);

// Express Configuration
const app = express();
const port = 3080;

app.use(bodyParser.json());
app.use(cors());
app.use(morgan("dev"));

// Routing

// Primary Open AI Route
app.post("/", async (req, res) => {
  const { message } = req.body;
  
  const runIndex = async () => {
    try {
      const operandService = operandClient(
        OperandService,
        "fwcttj4mz7ag6mfhowsstpv8a487vmjl",
        "https://mcp.operand.ai"
      );

      const results = await operandService.search({
        query: `${message}`,
        parentId: "6zosj9kzti9it76j", // Search over files in Brockport Admissions Folder
        maxResults: 3,
      });
  
      console.log("Operand Main:")
      for (const match of results.matches) {
        console.log("- " + match.snippet);
      }
  
      if (results) {
        return results.matches.map((m) => `- ${m.content}`).join("\n");
      } else {
        return "";
      }
    } catch (error) {
      console.log(error);
    }
  };

  let operandSearch = await runIndex(message);

  const emails = async () => {
    try {
      const operandService = operandClient(
        OperandService,
        "fwcttj4mz7ag6mfhowsstpv8a487vmjl",
        "https://mcp.operand.ai"
      );

      const results = await operandService.search({
        query: `${message}`,
        parentId: "edpefrppmgwy4utc", // Search over files in Brockport Admissions Folder
        maxResults: 1,
      });

      console.log("\nOperand Emails:")
      for (const match of results.matches) {
        console.log("- " + match.snippet);
      }

      if (results) {
        return results.matches.map((m) => `- ${m.content}`).join("\n");
      } else {
        return "";
      }
    } catch (error) {
      console.log(error);
    }
  };

  let operandEmail = await emails(message);

  const main = async() => {
    const completion = await openai.createChatCompletion({
      model: "gpt-3.5-turbo",
      messages: [{role: "system", content: "You are a helpful chatbot for SUNY Brockport. Use only the information provided to answer a question. If you do not know the answer to a question, refer the User to the Brockport website. If the user asks a question unrelated to SUNY Brockport, do not answer."}, 
      {role: "user", content: `Tell me about Shakespeare.`},
      {role: "assistant", content: `I'm sorry, as the chatbot for SUNY Brockport, I can only answer questions related to SUNY Brockport`},
      {role: "user", content: "I'm required to get a physical for a sport I play, where can I get that done?"},
      {role: "assistant", content: `I'm not sure how to answer that question, perhaps the SUNY Brockport website has resources to help https://www.brockport.edu/`},
      {role: "user", content: `Here is context for how to respond, follow these steps.\n 
      Try to answer enthusiastically, but be as clear as possible.\n
      Use only the following information to answer the question: ${operandSearch}\n
      For specific questions refer the User to the relevant website from the list: ${operandEmail}\n\n
      The Users question is: ${message}`}]
    });

    console.log(`\nQuestion:\n${message}\n\nAnswer:\n${completion.data.choices[0].message.content}\n\nTokens Used: ${completion.data.usage.total_tokens}`);

    res.json({
      message: completion.data.choices[0].message.content,
    });
  }
  main();
});

// Get Models Route

// Start the server
app.listen(port, () => {
  console.log(`server running`);
});

//module.exports = app;

export default app;
