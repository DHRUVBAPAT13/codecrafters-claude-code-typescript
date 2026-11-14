import OpenAI from "openai";
import fs from "node:fs"

const TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "Read",
      description: "Read and return the contents of a file",
      parameters: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "The path to the file to read",
          },
        },
        required: ["file_path"],
      },
    },
  },
];

async function main() {

	const [, , flag, prompt] = process.argv;
	const apiKey = process.env.OPENROUTER_API_KEY;
	const baseURL =
		process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";

	if (!apiKey) {
		throw new Error("OPENROUTER_API_KEY is not set");
	}
	if (flag !== "-p" || !prompt) {
		throw new Error("error: -p flag is required");
	}

	const client = new OpenAI({
		apiKey: apiKey,
		baseURL: baseURL,
	});

	const response = await client.chat.completions.create({
		model: "anthropic/claude-haiku-4.5",
		messages: [{ role: "user", content: prompt }],
		tools: TOOLS,
	});

	if (!response.choices || response.choices.length === 0) {
		throw new Error("no choices in response");
	}

	const message = response.choices[0].message;

	if(message.tool_calls && message.tool_calls.length > 0) {
		const toolCall = message.tool_calls[0];
		if(toolCall.type == "function" && toolCall.function.name == "Read") {
			const rawArgs = toolCall.function.arguments;
			const args = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;
			const content = fs.readFileSync(args.file_path, "utf-8");
			process.stdout.write(content);
		}
	}
	else if(message.content) {
		console.log(message.content);
	}
}

main();
