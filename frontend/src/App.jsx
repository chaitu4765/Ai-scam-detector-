import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, AlertTriangle, CheckCircle, Smartphone, Mail, Globe, Upload } from 'lucide-react'
import axios from 'axios'

// Configure Axios
const api = axios.create({
  baseURL: 'http://localhost:8000',
});

function App() {
  const [inputText, setInputText] = useState('')
  const [activeTab, setActiveTab] = useState('text') // text, url, qr
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleScan = async () => {
    setLoading(true);
    setResult(null);

    try {
      let response;
      if (activeTab === 'qr') {
        // Handle File Upload
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput && fileInput.files[0]) {
          const formData = new FormData();
          formData.append("file", fileInput.files[0]);
          response = await api.post('/decode/qr', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
        } else {
          setLoading(false);
          alert("Please select a QR code image.");
          return;
        }
      } else {
        // Handle Text/URL
        if (!inputText) {
          setLoading(false);
          return;
        };
        response = await api.post('/predict/text', { content: inputText, type: activeTab });
      }

      setTimeout(() => { // Mock delay for effect
        setResult(response.data);
        setLoading(false);
      }, 1000)

    } catch (error) {
      console.error("Scan failed", error);
      setLoading(false);
      // Fallback mock result if backend isn't running yet for demo
      setResult({
        is_phishing: true,
        confidence: 0.88,
        analysis: "Error connecting to backend, but here is a demo alert."
      })
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary selection:text-white">
      {/* Navbar */}
      <nav className="p-6 flex justify-between items-center border-b border-white/10 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2 text-primary font-bold text-xl tracking-tighter">
          <Shield className="w-8 h-8" />
          <span>PhishGuard AI</span>
        </div>
        <div className="hidden md:flex gap-6 text-sm font-medium text-muted-foreground">
          <a href="#" className="hover:text-primary transition-colors">Dashboard</a>
          <a href="#" className="hover:text-primary transition-colors">Live Map</a>
          <a href="#" className="hover:text-primary transition-colors">API</a>
        </div>
        <button className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-full text-sm font-bold transition-transform hover:scale-105">
          Log In
        </button>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center p-6 relative overflow-hidden">
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[128px] pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[128px] pointer-events-none" />

        <div className="max-w-4xl w-full flex flex-col items-center z-10">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl font-extrabold text-center tracking-tight mb-6 bg-gradient-to-br from-white to-gray-400 bg-clip-text text-transparent"
          >
            AI-Powered Threat Detection
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-lg md:text-xl text-muted-foreground text-center mb-12 max-w-2xl"
          >
            Secure your digital life. Detect phishing emails, malicious URLs, and fraudulent QR codes instantly with our advanced machine learning models.
          </motion.p>

          {/* Application Logic Wrapper */}
          <div className="w-full max-w-2xl bg-white/5 border border-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl">
            {/* Tabs */}
            <div className="flex gap-2 mb-6 p-1 bg-black/20 rounded-xl w-fit mx-auto">
              {['text', 'url', 'qr'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab ? 'bg-primary text-white shadow-lg' : 'text-muted-foreground hover:text-white'}`}
                >
                  {tab === 'text' && <div className="flex items-center gap-2"><Mail className="w-4 h-4" /> Message</div>}
                  {tab === 'url' && <div className="flex items-center gap-2"><Globe className="w-4 h-4" /> URL</div>}
                  {tab === 'qr' && <div className="flex items-center gap-2"><Smartphone className="w-4 h-4" /> QR Code</div>}
                </button>
              ))}
            </div>

            {/* Input Area */}
            <div className="relative mb-6">
              {activeTab === 'qr' ? (
                <div className="border-2 border-dashed border-white/20 rounded-xl h-48 flex flex-col items-center justify-center text-muted-foreground hover:border-primary/50 hover:bg-primary/5 transition-colors cursor-pointer group">
                  <Upload className="w-10 h-10 mb-4 text-white/50 group-hover:text-primary transition-colors" />
                  <p>Drag & Drop QR Code Image</p>
                  <p className="text-xs mt-2">or click to browse</p>
                  <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" />
                </div>
              ) : (
                <textarea
                  placeholder={activeTab === 'url' ? "Paste any suspicious link here..." : "Paste email content or SMS message here..."}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full h-32 bg-black/40 border border-white/10 rounded-xl p-4 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                ></textarea>
              )}
            </div>

            {/* Action Button */}
            <button
              onClick={handleScan}
              disabled={loading || (!inputText && activeTab !== 'qr')}
              className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-primary/25 transition-all active:scale-[0.98]"
            >
              {loading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                >
                  <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full" />
                </motion.div>
              ) : (
                <>
                  <span>Analyze content</span>
                  <Shield className="w-5 h-5" />
                </>
              )}
            </button>
          </div>

          {/* Results Display */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className={`mt-8 p-6 rounded-2xl border w-full max-w-2xl backdrop-blur-xl shadow-2xl ${result.is_phishing
                  ? 'bg-red-500/10 border-red-500/50 shadow-red-500/10'
                  : 'bg-green-500/10 border-green-500/50 shadow-green-500/10'
                  }`}
              >
                <div className="flex gap-4 items-start">
                  <div className={`p-3 rounded-full ${result.is_phishing ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                    {result.is_phishing ? <AlertTriangle className="w-8 h-8" /> : <CheckCircle className="w-8 h-8" />}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={`text-2xl font-bold ${result.is_phishing ? 'text-red-400' : 'text-green-400'}`}>
                        {result.is_phishing ? 'Threat Detected' : 'Safe Content'}
                      </h3>
                      <span className="text-sm font-mono opacity-70">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      {result.analysis}
                    </p>
                    {result.is_phishing && (
                      <div className="text-xs bg-red-500/10 border border-red-500/20 p-3 rounded-lg text-red-300">
                        Recommendation: Do not click any links. Delete this message immediately.
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <footer className="p-6 text-center text-xs text-muted-foreground">
        © 2025 PhishGuard AI. All rights reserved.
      </footer>
    </div>
  )
}

export default App
