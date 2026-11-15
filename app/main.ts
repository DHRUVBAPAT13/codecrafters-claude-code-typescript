import OpenAI from "openai";
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { pathToFileURL, fileURLToPath } from "node:url";
import { LSPClient } from "./lsp_client";

interface Skill {
    name: string;
    description: string;
    context?: string;
    body: string;
}

interface DiagnosticItem {
    line: number;
    character: number;
    message: string;
}

interface LspState {
    documentVersions: Map<string, number>;
    notificationCursor: number;
    lastShownDiagnostics: Map<string, Set<string>>;
}

const BASE_TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
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
    {
        type: "function",
        function: {
            name: "Bash",
            description: "Execute a Shell command",
            parameters: {
                type: "object",
                required: ["command"],
                properties: {
                    command: {
                        type: "string",
                        description: "The Shell command to execute",
                    },
                },
            },
        },
    },
    {
        type: "function",
        function: {
            name: "Skill",
            description: "Load a skill's instructions into the conversation",
            parameters: {
                type: "object",
                required: ["name"],
                properties: {
                    name: {
                        type: "string",
                        description: "The name of the skill to use",
                    },
                    args: {
                        type: "string",
                        description: "Optional arguments for the skill",
                    },
                },
            },
        },
    },
];

const LSP_TOOL: OpenAI.Chat.Completions.ChatCompletionTool = {
    type: "function",
    function: {
        name: "LSP",
        description:
            "Find symbol definitions (goToDefinition) and references (findReferences). Line and character inputs are 1-based.",
        parameters: {
            type: "object",
            properties: {
                operation: {
                    type: "string",
                    enum: ["goToDefinition", "findReferences"],
                },
                filePath: {
                    type: "string",
                    description:
                        "Absolute path, or a path relative to the agent's current working directory",
                },
                line: {
                    type: "integer",
                    minimum: 1,
                    description: "1-based line number of symbol",
                },
                character: {
                    type: "integer",
                    minimum: 1,
                    description:
                        "1-based character offset in UTF-16 code units (same as character count for ASCII)",
                },
            },
            required: ["operation", "filePath", "line", "character"],
        },
    },
};

function parseSkillFile(filePath: string, fallbackName: string): Skill | null {

    if (!fs.existsSync(filePath)) return null;

    const raw = fs.readFileSync(filePath, "utf-8");
    const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);

    if (!match) {
        return {
            name: fallbackName,
            description: "",
            body: raw.trim(),
        };
    }

    const frontmatter = match[1];
    const body = match[2].trim();

    let name = fallbackName;
    let description = "";
    let context: string | undefined;

    for (const line of frontmatter.split(/\r?\n/)) {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) continue;

        const key = line.slice(0, colonIdx).trim();
        const value = line
            .slice(colonIdx + 1)
            .trim()
            .replace(/^["']|["']$/g, "");

        if (key === "name" && value) name = value;
        if (key === "description") description = value;
        if (key === "context") context = value;
    }

    return { name, description, context, body };
}

function loadSkills(): Map<string, Skill> {

    const skills = new Map<string, Skill>();
    const skillsDir = path.join(process.cwd(), ".claude", "skills");

    if (!fs.existsSync(skillsDir)) return skills;

    for (const entry of fs.readdirSync(skillsDir, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;

        const skillFile = path.join(skillsDir, entry.name, "SKILL.md");
        const parsed = parseSkillFile(skillFile, entry.name);

        if (parsed) {
            skills.set(entry.name, parsed);
            skills.set(parsed.name, parsed);
        }
    }

    return skills;
}

function resolveSkill(
    skillName: string,
    skillsMap: Map<string, Skill>
): Skill | null {

    if (!skillName || skillName.includes("/") || skillName.includes("\\")) {
        return null;
    }

    return (
        skillsMap.get(skillName) ??
        parseSkillFile(
            path.join(process.cwd(), ".claude", "skills", skillName, "SKILL.md"),
            skillName
        )
    );
}

function expandStackedSkill(
    prompt: string,
    skillsMap: Map<string, Skill>
): { expandedSkills: Skill[]; rawArgs: string } {

    const expandedSkills: Skill[] = [];
    let remaining = prompt.trimStart();

    while (remaining.startsWith("/")) {
        const spaceIdx = remaining.search(/\s/);
        const token =
            spaceIdx === -1 ? remaining.slice(1) : remaining.slice(1, spaceIdx);
        const matched = resolveSkill(token, skillsMap);

        if (!matched || matched.context === "fork") {
            break;
        }

        expandedSkills.push(matched);
        remaining = spaceIdx === -1 ? "" : remaining.slice(spaceIdx).trimStart();
    }
    return { expandedSkills, rawArgs: remaining.trim() };
}

function substituteSkillArgs(body: string, rawArgs: string): string {

    const args = rawArgs ? rawArgs.split(/\s+/) : [];

    return body
        .replace(/\$ARGUMENTS\[(\d+)\]/g, (_, idx) => args[Number(idx)] ?? "")
        .replace(/\$ARGUMENTS\b/g, rawArgs)
        .replace(/\$(\d+)\b/g, (_, idx) => args[Number(idx)] ?? "");
}

async function ensureDocumentOpen(
    lspClient: LSPClient,
    absPath: string,
    uri: string,
    documentVersions: Map<string, number>
): Promise<void> {

    if (documentVersions.has(uri)) return;

    const ext = path.extname(absPath);
    const languageId = lspClient.extensionToLanguage?.[ext] ?? "plaintext";
    const text = fs.readFileSync(absPath, "utf-8");

    await lspClient.notify("textDocument/didOpen", {
        textDocument: {
            uri,
            languageId,
            version: 1,
            text,
        },
    });
    documentVersions.set(uri, 1);
}

async function syncWrittenDocument(
    lspClient: LSPClient,
    absPath: string,
    text: string,
    documentVersions: Map<string, number>
): Promise<void> {
    const ext = path.extname(absPath);
    const languageId = lspClient.extensionToLanguage?.[ext];
    if (!languageId) return;

    const uri = pathToFileURL(absPath).href;
    const currentVersion = documentVersions.get(uri);

    if (currentVersion === undefined) {
        await lspClient.notify("textDocument/didOpen", {
            textDocument: {
                uri,
                languageId,
                version: 1,
                text,
            },
        });
        documentVersions.set(uri, 1);
    } else {
        const nextVersion = currentVersion + 1;
        await lspClient.notify("textDocument/didChange", {
            textDocument: {
                uri,
                version: nextVersion,
            },
            contentChanges: [{ text }],
        });
        documentVersions.set(uri, nextVersion);
    }
}

function formatDiagnosticKey(diag: DiagnosticItem): string{

    return `${diag.line}:${diag.character}:${diag.message}`
}

async function drainNewDiagnostics(
    lspClient: LSPClient,
    lspState: LspState
): Promise<string> {

    const res = (await lspClient.notifications(lspState.notificationCursor, 0)) as {
        cursor?: number;
        notifications?: Array<{
            method?: string;
            params?: {
                uri?: string;
                version?: number;
                diagnostics?: Array<{
                    range?: { start?: { line?: number; character?: number; } };
                    message?: string;
                }>;
            };
        }>;
    };

    if (typeof res?.cursor === "number") {
        lspState.notificationCursor = res.cursor;
    }

    const notifications = Array.isArray(res?.notifications) ? res.notifications : [];

    const latestCurrentByUri = new Map<string, DiagnosticItem[]>();

    for (const notif of notifications) {
        if (notif?.method !== "textDocument/publishDiagnostics") continue;
        const params = notif.params;
        if (!params?.uri) continue;

        const recordedVersion = lspState.documentVersions.get(params.uri);
        if (
            typeof params.version === "number" &&
            recordedVersion !== undefined &&
            params.version !== recordedVersion
        ) {
            continue;
        }

        const items: DiagnosticItem[] = [];
        for (const d of params.diagnostics ?? []) {
            if (!d?.range?.start || typeof d.message !== "string") continue;
            items.push({
                line: Number(d.range.start.line) + 1,
                character: Number(d.range.start.character) + 1,
                message: d.message,
            });
        }
        latestCurrentByUri.set(params.uri, items);
    }

    if (latestCurrentByUri.size === 0) {
        return "";
    }

    const newByFile = new Map<string, string[]>();

    for (const [uri, currentItems] of latestCurrentByUri.entries()) {
        const prevShown =
            lspState.lastShownDiagnostics.get(uri) ?? new Set<string>();
        const nextShown = new Set<string>();
        const freshLines: string[] = [];

        for (const item of currentItems) {
            const key = formatDiagnosticKey(item);
            nextShown.add(key);
            if (!prevShown.has(key)) {
                freshLines.push(
                    `  [Line ${item.line}:${item.character}] ${item.message}`
                );
            }
        }

        lspState.lastShownDiagnostics.set(uri, nextShown);

        if (freshLines.length > 0) {
            const absPath = fileURLToPath(uri);
            const relPath =
                path.relative(process.cwd(), absPath) ||
                path.basename(absPath);
            newByFile.set(relPath, freshLines);
        }
    }

    if (newByFile.size === 0) {
        return "";
    }

    const sections: string[] = [];
    for (const [relPath, lines] of newByFile.entries()) {
        sections.push(`${relPath}:\n${lines.join("\n")}`);
    }

    return `<new-diagnostics>The following new diagnostic issues were detected:\n\n${sections.join("\n\n")}</new-diagnostics>`;
}

async function handleLspTool(
    args: Record<string, unknown>,
    lspClient: LSPClient | null,
    lspState: LspState
): Promise<string> {

    if (!lspClient) {
        return "error: LSP client is not configured";
    }

    const operation = String(args.operation ?? "");
    const rawFilePath = String(args.filePath ?? "");
    const line = Number(args.line);
    const character = Number(args.character);

    try {
        const absPath = path.resolve(process.cwd(), rawFilePath);
        const uri = pathToFileURL(absPath).href;

        if (operation === "goToDefinition") {
            await ensureDocumentOpen(lspClient, absPath, uri, lspState.documentVersions);

            const rawResponse = await lspClient.request(
                "textDocument/definition",
                {
                    textDocument: { uri },
                    position: {
                        line: line - 1,
                        character: character - 1,
                    },
                }
            );

            const location = Array.isArray(rawResponse)
                ? rawResponse[0]
                : rawResponse;

            if (!location) {
                return "No definition found";
            }

            const targetUri: string = location.uri ?? location.targetUri;
            const range = location.range ?? location.targetSelectionRange;

            if (!targetUri || !range?.start) {
                return "No definition found";
            }

            const targetAbsPath = fileURLToPath(targetUri);
            const relPath =
                path.relative(process.cwd(), targetAbsPath) ||
                path.basename(targetAbsPath);
            const defLine = Number(range.start.line) + 1;
            const defChar = Number(range.start.character) + 1;

            return `Defined in ${relPath}:${defLine}:${defChar}`;
        }

        if (operation === "findReferences") {
            await ensureDocumentOpen(lspClient, absPath, uri, lspState.documentVersions);

            const rawResponse = await lspClient.request(
                "textDocument/references",
                {
                    textDocument: { uri },
                    position: {
                        line: line - 1,
                        character: character - 1,
                    },
                    context: {
                        includeDeclaration: true,
                    },
                }
            );

            const locations = Array.isArray(rawResponse) ? rawResponse : [];
            if (locations.length === 0) {
                return "No references found";
            }

            const grouped = new Map<string, string[]>();
            let totalCount = 0;

            for (const loc of locations) {
                const targetUri: string | undefined = loc?.uri ?? loc?.targetUri;
                const range = loc?.range ?? loc?.targetSelectionRange;
                if (!targetUri || !range?.start) continue;

                const targetAbsPath = fileURLToPath(targetUri);
                const relPath =
                    path.relative(process.cwd(), targetAbsPath) ||
                    path.basename(targetAbsPath);
                const refLine = Number(range.start.line) + 1;
                const refChar = Number(range.start.character) + 1;

                if (!grouped.has(relPath)) {
                    grouped.set(relPath, []);
                }
                grouped.get(relPath)!.push(`  Line ${refLine}:${refChar}`);
                totalCount++;
            }

            if (totalCount === 0) {
                return "No references found";
            }

            const sections: string[] = [];
            for (const [file, lines] of grouped.entries()) {
                sections.push(`${file}:\n${lines.join("\n")}`);
            }

            return `Found ${totalCount} references across ${grouped.size} files:\n\n${sections.join("\n\n")}`;
        }

        return `error: unsupported LSP operation '${operation}'`;
    } catch (err) {
        return `error: ${(err as Error).message}`;
    }
}

async function executeTool(
    name: string,
    rawArgs: string | Record<string, unknown>,
    skillsMap: Map<string, Skill>,
    client: OpenAI,
    tools: OpenAI.Chat.Completions.ChatCompletionTool[],
    lspClient: LSPClient | null,
    lspState: LspState
): Promise<string> {

    const args = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;

    switch (name) {
        case "Read": {
            try {
                return fs.readFileSync(String(args.file_path), "utf-8");
            } catch (err) {
                return `error: ${(err as Error).message}`;
            }
        }
        case "Write": {
            try {
                const filePath = String(args.file_path);
                const content = String(args.content ?? "");
                const absPath = path.resolve(process.cwd(), filePath);

                fs.mkdirSync(path.dirname(absPath), { recursive: true });
                fs.writeFileSync(absPath, content, "utf-8");

                if (lspClient) {
                    await syncWrittenDocument(
                        lspClient,
                        absPath,
                        content,
                        lspState.documentVersions
                    );
                }

                return `Successfully wrote ${Buffer.byteLength(content)} bytes to ${filePath}`;
            } catch (err) {
                return `error: ${(err as Error).message}`;
            }
        }
        case "Bash": {
            try {
                return execSync(String(args.command), {
                    encoding: "utf-8",
                    stdio: ["ignore", "pipe", "pipe"],
                });
            } catch (err: any) {
                const stdout = err.stdout ? String(err.stdout) : "";
                const stderr = err.stderr ? String(err.stderr) : "";
                return `${stdout}${stderr}` || `error: ${err.message}`;
            }
        }
        case "Skill": {
            const skillName = String(args.name ?? "");
            const skillArgs = args.args ? String(args.args) : "";
            const matched = resolveSkill(skillName, skillsMap);
            if (!matched) {
                return `error: unknown skill '${skillName}'`;
            }
            const body = substituteSkillArgs(matched.body, skillArgs);

            if (matched.context === "fork") {
                console.error(`[Subagent] Forking skill '${matched.name}'`);
                const subagentMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] =
                    [{ role: "user", content: body }];
                const subagentResult = await runAgentLoop(
                    client,
                    subagentMessages,
                    skillsMap,
                    tools,
                    lspClient,
                    lspState,
                );
                return `Skill ${matched.name} ran in a separate context and returned: ${subagentResult}`;
            }
            return body;
        }
        case "LSP": {
            return handleLspTool(args, lspClient, lspState);
        }
        default: {
            return `error: unknown tool '${name}'`;
        }
    }
}

async function runAgentLoop(
    client: OpenAI,
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    skillMap: Map<string, Skill>,
    tools: OpenAI.Chat.Completions.ChatCompletionTool[],
    lspClient: LSPClient | null,
    lspState: LspState
): Promise<string> {

    while (true) {

        if (lspClient) {
            const diagnosticsFeedback = await drainNewDiagnostics(
                lspClient,
                lspState
            );
            if (diagnosticsFeedback) {
                const lastMsg = messages[messages.length - 1];
                if (lastMsg && lastMsg.role === "tool") {
                    lastMsg.content = `${String(lastMsg.content)}\n\n${diagnosticsFeedback}`;
                } else {
                    messages.push({
                        role: "user",
                        content: diagnosticsFeedback,
                    });
                }
            }
        }

        const response = await client.chat.completions.create({
            model: "anthropic/claude-haiku-4.5",
            messages: messages,
            tools: tools,
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
                const result = await executeTool(
                    toolCall.function.name,
                    toolCall.function.arguments,
                    skillMap,
                    client,
                    tools,
                    lspClient,
                    lspState
                );

                messages.push({
                    role: "tool",
                    tool_call_id: toolCall.id,
                    content: result,
                });
            }
            continue;
        }

        return message.content ?? "";
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

    const lspClient = await LSPClient.fromEnvironment();
    if (lspClient) {
        console.error("LSP client connected");
    }

    const tools = lspClient ? [...BASE_TOOLS, LSP_TOOL] : BASE_TOOLS;
    const lspState: LspState = {
        documentVersions: new Map<string, number>(),
        notificationCursor: 0,
        lastShownDiagnostics: new Map<string, Set<string>>(),
    };

    const skillsMap = loadSkills();
    const uniqueSkills = Array.from(new Set(skillsMap.values()));

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

    if (uniqueSkills.length > 0) {
        const skillList = uniqueSkills
            .map((s) => `- ${s.name}: ${s.description}`)
            .join("\n");

        messages.push({
            role: "system",
            content: `You have access to the following skills:\n\n${skillList}\n\nIf a skill matches the user's request, call the Skill tool with its name\nand follow the instructions it returns.`,
        });
    }

    const { expandedSkills, rawArgs } = expandStackedSkill(prompt, skillsMap);

    if (expandedSkills.length > 0) {
        for (const skill of expandedSkills) {
            messages.push({
                role: "user",
                content: substituteSkillArgs(skill.body, rawArgs),
            });
        }
    } else {
        messages.push({ role: "user", content: prompt });
    }

    const client = new OpenAI({
        apiKey: apiKey,
        baseURL: baseURL,
    });

    const finalOutput = await runAgentLoop(
        client,
        messages,
        skillsMap,
        tools,
        lspClient,
        lspState
    );
    if (finalOutput) {
        console.log(finalOutput);
    }
}

main();