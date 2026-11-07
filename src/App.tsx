import React, { useState, useEffect, useRef } from 'react';
import { 
  AutoTokenizer, 
  AutoProcessor, 
  AutoModelForCausalLM, 
  AutoModelForVision2Seq, 
  TextStreamer, 
  InterruptableStoppingCriteria, 
  env 
} from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache = true;

type ChatMessage = { role: 'user' | 'assistant'; content: string };
type ModelInfo = { id: string; name: string; type: string; size: string; ram: string; best: string };
type ModelData = { model: any; processor: any; info: ModelInfo };

const modelCatalog = [
  { category: 'Mobile Optimized', items: [
    { id: 'onnx-community/Qwen2.5-0.5B-Instruct-ONNX', name: 'Qwen 2.5 (0.5B)', type: 'Text', size: '~350 MB', ram: '1.5 GB', best: 'Fast mobile chat' },
    { id: 'onnx-community/Llama-3.2-1B-Instruct-ONNX', name: 'Llama 3.2 (1B)', type: 'Text', size: '~700 MB', ram: '2.5 GB', best: 'Quality mobile chat' },
    { id: 'HuggingFaceTB/SmolVLM2-256M-ONNX', name: 'SmolVLM2 (256M)', type: 'Vision+Text', size: '~200 MB', ram: '1.5 GB', best: 'Ultra-light vision' }
  ]},
  { category: 'Desktop Optimized', items: [
    { id: 'onnx-community/Qwen3.5-2B-ONNX', name: 'Qwen 3.5 (2B)', type: 'Vision+Text', size: '~1.6 GB', ram: '6.0 GB', best: 'Balanced desktop' },
    { id: 'onnx-community/gemma-3-4b-it-ONNX', name: 'Gemma 3 (4B)', type: 'Vision+Text', size: '~2.8 GB', ram: '8.0 GB', best: 'Multilingual' },
    { id: 'onnx-community/Qwen2.5-7B-Instruct-ONNX', name: 'Qwen 2.5 (7B)', type: 'Text', size: '~4.5 GB', ram: '12.0 GB', best: 'Powerful reasoning' }
  ]},
  { category: 'Reasoning (Advanced)', items: [
    { id: 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX', name: 'DeepSeek R1 (1.5B)', type: 'Text', size: '~1.2 GB', ram: '4.0 GB', best: 'Logic & Coding' }
  ]}
];

export default function App() {
  const [screen, setScreen] = useState<'landing' | 'catalog' | 'loading' | 'chat'>('landing');
  const [model, setModel] = useState<ModelData | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const loadModel = async (modelInfo: ModelInfo) => {
    setScreen('loading');
    setLoadingProgress(0);
    try {
      const isVision = modelInfo.type.includes('Vision');
      const ModelClass = isVision ? AutoModelForVision2Seq : AutoModelForCausalLM;
      
      const proc = await (isVision ? AutoProcessor : AutoTokenizer).from_pretrained(modelInfo.id);
      
      const loadedModel = await ModelClass.from_pretrained(modelInfo.id, {
        dtype: isVision ? { embed_tokens: "q4", vision_encoder: "fp16", decoder_model_merged: "q4" } : "q4",
        device: navigator.gpu ? "webgpu" : "wasm",
        progress_callback: (progress: any) => {
          if (progress.status === 'progress') {
            setLoadingProgress((progress.loaded / progress.total) * 100);
          }
        }
      });
      setModel({ model: loadedModel, processor: proc, info: modelInfo });
      setScreen('chat');
    } catch (error) {
      console.error(error);
      alert("Failed to load model.");
      setScreen('catalog');
    }
  };

  const handleSend = async (text: string) => {
    if (!text || !model || isGenerating) return;
    setIsGenerating(true);
    const newHistory: ChatMessage[] = [...chatHistory, { role: 'user', content: text }];
    setChatHistory(newHistory);
    
    let assistantContent = '';
    const newHistoryWithAssistant: ChatMessage[] = [...newHistory, { role: 'assistant', content: assistantContent }];
    setChatHistory(newHistoryWithAssistant);
    
    try {
      const { model: loadedModel, processor } = model;
      const tokenizer = processor.tokenizer || processor;
      
      const prompt = tokenizer.apply_chat_template(newHistory, { add_generation_prompt: true, tokenize: false });
      const inputs = await processor(prompt);
      const stoppingCriteria = new InterruptableStoppingCriteria();
      
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: (token: string) => {
          assistantContent += token;
          setChatHistory(prev => [
            ...prev.slice(0, -1),
            { role: 'assistant', content: assistantContent }
          ]);
        }
      });
      
      await loadedModel.generate({ ...inputs, max_new_tokens: 1024, streamer, stopping_criteria: stoppingCriteria });
    } catch (error) {
      console.error(error);
    }
    setIsGenerating(false);
  };

  return (
    <div className="min-h-screen p-4 flex flex-col items-center">
      {screen === 'landing' && (
        <div className="glass-panel p-10 rounded-[var(--radius)] shadow-[var(--neu-out)] bg-[var(--surface)] text-center">
          <h1 className="text-3xl text-[var(--accent)] font-serif mb-4">☬SHΞN™ AI Chat</h1>
          <p className="text-[var(--text-dim)] mb-6">Offline AI in your browser.</p>
          <button onClick={() => setScreen('catalog')} className="shadow-[var(--neu-btn-out)] px-8 py-3 rounded-xl text-[var(--accent)] font-bold">Get Started</button>
        </div>
      )}
      
      {screen === 'catalog' && (
        <div className="w-full max-w-4xl">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-2xl font-serif text-[var(--text)]">Select a Model</h2>
            <button onClick={() => setScreen('landing')} className="text-[var(--text-dim)] hover:text-[var(--accent)]">Back</button>
          </div>
          {modelCatalog.map(group => (
            <div key={group.category}>
              <h3 className="text-sm text-[var(--text-dim)] uppercase mt-8 mb-4 tracking-widest">{group.category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {group.items.map(m => (
                  <button key={m.id} onClick={() => loadModel(m)} className="p-6 rounded-[var(--radius)] shadow-[var(--neu-out)] bg-[var(--surface)] text-left flex flex-col gap-3 hover:transform hover:-translate-y-1 transition-all">
                    <h4 className="text-lg text-[var(--text)]">{m.name}</h4>
                    <span className="text-xs text-[var(--accent)] font-mono">{m.type}</span>
                    <p className="text-sm text-[var(--text-dim)]">{m.best}</p>
                    <div className="text-xs text-[var(--text-dim)] flex justify-between mt-auto pt-4">
                      <span>⬇ {m.size}</span>
                      <span>RAM: {m.ram}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {screen === 'loading' && (
        <div className="glass-panel p-10 rounded-[var(--radius)] shadow-[var(--neu-out)] bg-[var(--surface)] w-full max-w-lg text-center">
          <h2 className="text-xl mb-4">Downloading Model...</h2>
          <div className="w-full h-4 bg-[var(--surface)] rounded-full shadow-[var(--neu-in)] overflow-hidden">
            <div className="h-full bg-[var(--accent)] transition-all duration-300" style={{ width: `${loadingProgress}%` }} />
          </div>
        </div>
      )}

      {screen === 'chat' && (
        <div className="w-full max-w-3xl flex flex-col h-[90vh]">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg text-[var(--text)]">{model?.info.name}</h3>
            <button onClick={() => { setChatHistory([]); setScreen('catalog'); }} className="text-[var(--text-dim)]">Reset</button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {chatHistory.map((msg, i) => (
              <div key={i} className={`p-5 rounded-2xl ${msg.role === 'user' ? 'bg-[var(--surface)] shadow-[var(--neu-out)] ml-auto text-[var(--accent)] max-w-[85%]' : 'bg-[var(--surface)] shadow-[var(--neu-in)] mr-auto text-[var(--text)] max-w-[85%]'}`}>
                {msg.content.split('\n').map((line, j) => <p key={j} className="mb-1">{line}</p>)}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
          <div className="p-4 bg-[var(--surface)] rounded-2xl shadow-[var(--neu-out)] flex gap-4">
            <input type="text" className="flex-1 bg-transparent outline-none p-2 text-[var(--text)]" placeholder="Type a message..." onKeyDown={(e) => { 
              if(e.key === 'Enter') { 
                const target = e.target as HTMLInputElement;
                handleSend(target.value); 
                target.value = ''; 
              } 
            }} />
          </div>
        </div>
      )}
    </div>
  );
}
