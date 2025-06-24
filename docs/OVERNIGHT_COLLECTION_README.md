# 🌙 Overnight Research Collection System

Automated system to collect real academic papers from Google Scholar while you sleep.

## 🚀 Quick Start

### Tonight (Before Sleep):
```bash
python start_overnight_collection.py
```

### Tomorrow Morning (When You Wake Up):
```bash
# Stop the collector with Ctrl+C
# Then run:
python morning_collection_summary.py
```

## 📊 What It Does

- **Searches 8 research topics** with multiple queries each
- **Collects 200-400 papers** overnight (8 hours)
- **Respectful delays**: 15-30s between papers, 30-60s between queries
- **Auto-saves progress** every 10 papers
- **Updates your research platform** automatically

## ⏰ Collection Schedule

```
🔄 Collection Cycle (repeats all night):
├── Scientific Revolutions (8 queries)
├── Enlightenment (8 queries) 
├── Decolonization (8 queries)
├── Haitian Revolution (8 queries)
├── Tokugawa Japan (8 queries)
├── Ming/Qing Dynasties (8 queries)
├── Early Modern Exploration (8 queries)
└── Southeast Asia Colonialism (8 queries)

⏸️ Delays:
├── Between papers: 15-30 seconds
├── Between queries: 30-60 seconds  
├── Between topics: 60-120 seconds
└── Between cycles: 5-10 minutes
```

## 📁 Files Created

- `overnight_collection_results_YYYYMMDD_HHMM.json` - Final results
- `overnight_collection_progress_YYYYMMDD_HHMM.json` - Progress saves
- Updated `webapp/data/research_results.json` - Your platform data

## 🛑 How to Stop

**Press `Ctrl+C` when you wake up** - the system will:
1. Save all collected data
2. Update your research platform  
3. Generate a summary report

## 📊 Expected Results (8 hours)

- **Papers**: 200-400 high-quality academic papers
- **Citations**: 10,000+ total citations
- **Topics**: All 8 research areas covered
- **Quality**: Real papers with abstracts, authors, citation counts

## 🌐 View Results

After collection, visit: http://127.0.0.1:5000

Your research platform will be updated with all new papers!

## 🔧 Troubleshooting

### If Collection Stops Early:
- Run `python morning_collection_summary.py` to process what was collected
- Check for error messages in the terminal

### If No Results:
- Ensure `scholarly` library is installed: `pip install scholarly`
- Check internet connection
- Try running a shorter test first

### If Rate Limited:
- The system automatically handles rate limiting
- Longer delays will be applied automatically
- Collection will continue when limits reset

## 💡 Tips

1. **Start before bed** - Let it run 6-8 hours for best results
2. **Keep computer awake** - Disable sleep mode for the night
3. **Stable internet** - Ensure good connection for overnight collection
4. **Check in morning** - Run summary script to see what was collected

## 🎯 Collection Strategy

The system uses intelligent search strategies:
- **Diverse queries** for each topic
- **Random query selection** to avoid repetition  
- **Quality filtering** - only keeps papers with good metadata
- **Citation-based ranking** - prioritizes highly-cited papers
- **Deduplication** - avoids collecting the same paper twice

## 🌅 Morning Workflow

1. **Wake up** ☕
2. **Press Ctrl+C** to stop collection 🛑
3. **Run summary script** 📊
4. **Check your platform** 🌐
5. **Enjoy new research!** 🎉

---

**Sweet dreams! Your research platform will be much richer in the morning! 🌙✨**
