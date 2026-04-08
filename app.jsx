import React, { useMemo, useRef, useState } from "react";
import { pipeline } from "@huggingface/transformers";
import { Send, Trash2, Loader2, Cpu, Bot } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";

const MODEL_ID = "onnx-community/Llama-3.2-1B-Instruct";
const DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant.";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi. Load the model, then send a message to start chatting.",
    },
  ]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [status, setStatus] = useState("Model not loaded yet.");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");

  const generatorRef = useRef(null);
  const generatorPromiseRef = useRef(null);

  const webgpuAvailable = useMemo(() => typeof navigator !== "undefined" && !!navigator.gpu, []);

  async function getGenerator() {
    if (generatorRef.current) return generatorRef.current;
    if (generatorPromiseRef.current) return generatorPromiseRef.current;

    setError("");
    setIsLoadingModel(true);
    setStatus(`Loading ${MODEL_ID}… the first run downloads the model and may take a while.`);

    generatorPromiseRef.current = pipeline("text-generation", MODEL_ID)
      .then((generator) => {
        generatorRef.current = generator;
        setStatus("Model ready.");
        return generator;
      })
      .catch((err) => {
        generatorPromiseRef.current = null;
        throw err;
      })
      .finally(() => {
        setIsLoadingModel(false);
      });

    return generatorPromiseRef.current;
  }

  async function loadModel() {
    try {
      await getGenerator();
    } catch (err) {
      console.error(err);
      setError(err?.message || "Failed to load the model.");
      setStatus("Model load failed.");
    }
  }

  function clearChat() {
    setMessages([
      {
        role: "assistant",
        content: "Chat cleared. Send a message when you are ready.",
      },
    ]);
    setError("");
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || isGenerating || isLoadingModel) return;

    setError("");
    setInput("");

    const userMessage = { role: "user", content: text };
    const placeholder = { role: "assistant", content: "Thinking…", pending: true };

    const visibleHistory = [...messages, userMessage]
      .filter((message) => !message.pending)
      .map(({ role, content }) => ({ role, content }));

    const conversation = [
      { role: "system", content: systemPrompt.trim() || DEFAULT_SYSTEM_PROMPT },
      ...visibleHistory,
    ];

    setMessages((prev) => [...prev, userMessage, placeholder]);
    setIsGenerating(true);
    setStatus("Generating response…");

    try {
      const generator = await getGenerator();
      const output = await generator(conversation, {
        max_new_tokens: 256,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.05,
      });

      const generated = output?.[0]?.generated_text;
      const assistantText = Array.isArray(generated)
        ? generated.at(-1)?.content || ""
        : typeof generated === "string"
          ? generated
          : "";

      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.findLastIndex((message) => message.pending);
        if (lastIndex >= 0) {
          next[lastIndex] = {
            role: "assistant",
            content: assistantText || "No response returned.",
          };
        }
        return next;
      });

      setStatus("Ready.");
    } catch (err) {
      console.error(err);
      const message = err?.message || "Generation failed.";
      setError(message);
      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.findLastIndex((item) => item.pending);
        if (lastIndex >= 0) {
          next[lastIndex] = {
            role: "assistant",
            content: `Error: ${message}`,
          };
        }
        return next;
      });
      setStatus("Generation failed.");
    } finally {
      setIsGenerating(false);
    }
  }

  function onComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-[320px_1fr]">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle className="text-xl">Transformers.js chat</CardTitle>
            <CardDescription>
              A browser-only SPA for {MODEL_ID} using conversation-style messages.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary" className="rounded-full px-3 py-1">
                <Bot className="mr-1 h-3.5 w-3.5" />
                {MODEL_ID}
              </Badge>
              <Badge variant="outline" className="rounded-full px-3 py-1">
                <Cpu className="mr-1 h-3.5 w-3.5" />
                device not specified
              </Badge>
              <Badge variant="outline" className="rounded-full px-3 py-1">
                WebGPU available: {webgpuAvailable ? "yes" : "no"}
              </Badge>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">System prompt</label>
              <Textarea
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                className="min-h-[140px] rounded-2xl"
              />
            </div>

            <div className="rounded-2xl border bg-white p-3 text-sm text-slate-600">
              <div className="font-medium text-slate-900">Status</div>
              <div className="mt-1">{status}</div>
              <div className="mt-2 text-xs text-slate-500">
                The app does not pass a <code>device</code> option, so runtime selection is left to the library/browser.
              </div>
            </div>

            {error ? (
              <div className="rounded-2xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                {error}
              </div>
            ) : null}

            <div className="flex flex-col gap-2 sm:flex-row">
              <Button
                onClick={loadModel}
                disabled={isLoadingModel || !!generatorRef.current}
                className="rounded-2xl"
              >
                {isLoadingModel ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                {generatorRef.current ? "Model loaded" : "Load model"}
              </Button>
              <Button variant="outline" onClick={clearChat} className="rounded-2xl">
                <Trash2 className="mr-2 h-4 w-4" />
                Clear chat
              </Button>
            </div>

            <p className="text-xs leading-5 text-slate-500">
              First load can be large and memory-intensive because the model runs entirely in the browser.
            </p>
          </CardContent>
        </Card>

        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle className="text-xl">Conversation</CardTitle>
            <CardDescription>
              Press Enter to send. Use Shift+Enter for a new line.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex h-[75vh] flex-col gap-4">
            <ScrollArea className="flex-1 rounded-2xl border bg-white p-4">
              <div className="space-y-4">
                {messages.map((message, index) => {
                  const isAssistant = message.role === "assistant";
                  return (
                    <div
                      key={`${message.role}-${index}`}
                      className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm ${
                          isAssistant
                            ? "bg-slate-100 text-slate-900"
                            : "bg-slate-900 text-slate-50"
                        }`}
                      >
                        <div className="mb-1 text-[11px] uppercase tracking-wide opacity-70">
                          {message.role}
                        </div>
                        <div className="whitespace-pre-wrap leading-6">{message.content}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>

            <div className="flex gap-3">
              <Input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder="Type your message..."
                className="rounded-2xl"
                disabled={isGenerating || isLoadingModel}
              />
              <Button
                onClick={() => void sendMessage()}
                disabled={!input.trim() || isGenerating || isLoadingModel}
                className="rounded-2xl"
              >
                {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
