import { operandClient, OperandService } from "@operandinc/sdk";

export const OperandPortion = async (
  prompt: string,
) => {
    let operand = '';

    try {
        const operandService = operandClient(
            OperandService,
            "fwcttj4mz7ag6mfhowsstpv8a487vmjl",
            "https://mcp.operand.ai"
        );

        const res = await operandService.search({
            query: prompt,
            parentId: "6zosj9kzti9it76j", // Search over files in Brockport Admissions Folder
            maxResults: 3,
        });

    operand = res.matches.map((match) => `- ${match.snippet}`).join("\n");

    } catch (error) {
        console.log(error);
    }

    const result = `Here is relevant information to the prompt:\n${operand}\n\nPrompt: ${prompt}`;

    return result;
};

