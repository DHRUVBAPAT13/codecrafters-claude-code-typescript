import OpenAI from "openai";
import fs from "node:fs"
import path from "node:path";

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
	{
		type: "function",
		function: {
			name: "Write",
			description: "Write a content to a file",
			parameters: {
				type: "object",
				required: ["file_path", "content"],
				properties: {
					file_path: {
						type: "string",
						description: "The path of the file to write to",
					},
					content: {
						type: "string",
						description: "The content to write to the file",
					},
				},
			},
		},
	},
];

function executeTool(
	name: string,
	rawArgs: string | Record<string, unknown>
): string {
	const args = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;

	switch (name) {
		case "Read": {
			try {
				return fs.readFileSync(String(args.file_path), "utf-8")
			} catch (err) {
				return `error: ${(err as Error).message}`;
			}
		}
		case "Write": {
			try {
				const filePath = String(args.file_path);
				const content = String(args.content ?? "");
				
				fs.mkdirSync(path.dirname(filePath), {recursive: true});
				fs.writeFileSync(filePath, content, "utf-8");
				return `Successfully wrote ${Buffer.byteLength(content)} bytes to {filePath}`;
			} catch (err) {
				return `error: ${err as Error}.message`;
			}
		}
		default: {
			return `error: unknown tool '${name}'`;
		}
	}
}

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

	const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
		{ role: "user", content: prompt },
	];

	while (true) {
		const response = await client.chat.completions.create({
			model: "anthropic/claude-haiku-4.5",
			messages: messages,
			tools: TOOLS,
		});

		if (!response.choices || response.choices.length === 0) {
			throw new Error("no choices in response");
		}

		const message = response.choices[0].message;
		messages.push(message);

		if (message.tool_calls && message.tool_calls.length > 0) {
			for (const toolCall of message.tool_calls) {
				if (toolCall.type !== "function") continue;

				console.error(`[Tool] Executing ${toolCall.function.name}...`);
				const result = executeTool(
					toolCall.function.name,
					toolCall.function.arguments,
				);

				messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: result,
				});
			}
			continue;
		}

		if (message.content) {
			console.log(message.content);
		}
		break;
	}
}

main();
