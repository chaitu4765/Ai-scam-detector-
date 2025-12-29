import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, AlertTriangle, CheckCircle, Smartphone, Mail, Globe, Upload } from 'lucide-react'
import axios from 'axios'

// Configure Axios
const api = axios.create({
  baseURL: import.meta.env.PROD ? '/api' : 'http://localhost:8000',
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

      const errorMessage = error.response
        ? `Error ${error.response.status}: ${error.response.data?.detail || error.response.statusText}`
        : error.message || "Network error. Please check your connection.";

      setResult({
        is_phishing: true,
        confidence: 0.88,
        analysis: `Error connecting to backend: ${errorMessage}`
      })
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary selection:text-white relative overflow-hidden">

      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-primary/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[120px]" />
        <div className="absolute top-[20%] right-[10%] w-[300px] h-[300px] bg-blue-500/10 rounded-full blur-[100px]" />
      </div>

      {/* Navbar */}
      <nav className="p-6 flex justify-between items-center z-50 relative">
        <div className="flex items-center gap-2 text-primary font-bold text-xl tracking-tighter cursor-default">
          <Shield className="w-8 h-8 drop-shadow-[0_0_10px_rgba(255,100,100,0.5)]" />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-white">PhishGuard AI</span>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center p-6 relative z-10">

        <div className="max-w-4xl w-full flex flex-col items-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-4">
              <span className="text-white drop-shadow-lg">DETECT</span><br />
              <span className="text-primary drop-shadow-[0_0_20px_rgba(255,100,100,0.5)]">PHISHING</span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground/80 font-medium max-w-2xl mx-auto">
              AI-powered security for your emails and URLs.
            </p>
          </motion.div>

          {/* Application Logic Wrapper */}
          <div className="w-full max-w-2xl bg-black/40 border border-white/5 backdrop-blur-2xl rounded-3xl p-8 shadow-[0_0_50px_rgba(0,0,0,0.5)]">
            {/* Tabs */}
            <div className="flex gap-2 mb-6 p-1 bg-white/5 rounded-xl w-fit mx-auto">
              {['text', 'url', 'qr'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${activeTab === tab
                    ? 'bg-primary text-white shadow-[0_0_20px_javax.swing.plaf.ColorUIResource[r=255,g=100,b=100]] transform scale-105'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                    }`}
                >
                  {tab === 'text' && <div className="flex items-center gap-2"><Mail className="w-4 h-4" /> Message</div>}
                  {tab === 'url' && <div className="flex items-center gap-2"><Globe className="w-4 h-4" /> URL</div>}
                  {tab === 'qr' && <div className="flex items-center gap-2"><Smartphone className="w-4 h-4" /> QR Code</div>}
                </button>
              ))}
            </div>

            {/* Input Area */}
            <div className="relative mb-8">
              {activeTab === 'qr' ? (
                <div className="border-2 border-dashed border-white/10 rounded-2xl h-48 flex flex-col items-center justify-center text-muted-foreground hover:border-primary/50 hover:bg-primary/5 transition-colors cursor-pointer group">
                  <Upload className="w-12 h-12 mb-4 text-white/30 group-hover:text-primary transition-colors" />
                  <p className="font-medium text-lg">Drag & Drop QR Code</p>
                  <p className="text-sm opacity-50 mt-1">or click to browse</p>
                  <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" />
                </div>
              ) : (
                <textarea
                  placeholder={activeTab === 'url' ? "Paste suspicious URL here (e.g., http://login-secure.com)..." : "Paste email content here to scan for threats..."}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full h-40 bg-black/50 border border-white/10 rounded-2xl p-6 text-lg text-white placeholder-white/20 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 resize-none transition-all shadow-inner"
                ></textarea>
              )}
            </div>

            {/* Action Button - Styled like "Get Started" */}
            <button
              onClick={handleScan}
              disabled={loading || (!inputText && activeTab !== 'qr')}
              className="w-full bg-[#FF4D4D] hover:bg-[#FF3333] disabled:opacity-50 disabled:cursor-not-allowed text-white text-lg font-bold py-5 rounded-2xl flex items-center justify-center gap-3 shadow-[0_10px_30px_rgba(255,77,77,0.4)] hover:shadow-[0_15px_40px_rgba(255,77,77,0.6)] transition-all transform hover:-translate-y-1 active:translate-y-0"
            >
              {loading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                >
                  <div className="w-6 h-6 border-4 border-white/30 border-t-white rounded-full" />
                </motion.div>
              ) : (
                <>
                  <span>SCAN NOW</span>
                  <div className="bg-white/20 p-1 rounded-full">
                    <Shield className="w-5 h-5 fill-current" />
                  </div>
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
                className={`mt-8 p-8 rounded-3xl border w-full max-w-2xl backdrop-blur-3xl shadow-2xl relative overflow-hidden ${result.is_phishing
                  ? 'bg-red-500/10 border-red-500/30'
                  : 'bg-green-500/10 border-green-500/30'
                  }`}
              >
                <div className="absolute inset-0 z-0 opacity-20 bg-gradient-to-b from-white/10 to-transparent pointer-events-none" />

                <div className="flex gap-6 items-center relative z-10">
                  <div className={`p-5 rounded-2xl shadow-lg border border-white/10 ${result.is_phishing ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}>
                    {result.is_phishing ? <AlertTriangle className="w-10 h-10" /> : <CheckCircle className="w-10 h-10" />}
                  </div>
                  <div className="flex-1">
                    <h3 className={`text-3xl font-black mb-1 ${result.is_phishing ? 'text-red-400' : 'text-green-400'}`}>
                      {result.is_phishing ? 'DANGEROUS' : 'SAFE'}
                    </h3>
                    <p className="text-lg text-white/80 font-medium">
                      {result.analysis}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm opacity-50 uppercase tracking-widest font-bold">Confidence</div>
                    <div className="text-4xl font-black">{(result.confidence * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

    </div>
  )
}

export default App
