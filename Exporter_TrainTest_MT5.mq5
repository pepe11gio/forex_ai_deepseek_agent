//+------------------------------------------------------------------+
//| Exporter_LiveAppend_MT5.mq5                                      |
//| EA: append to CSV on every newly closed candle                   |
//| Exports: time,open,high,low,close,volume,                        |
//|          EMA_20,EMA_50,RSI_14,ATR_14_PIPS,ADX_14,PLUS_DI_14,MINUS_DI_14
//+------------------------------------------------------------------+
#property strict

input string          InpSymbol        = "";            // Symbol (empty = chart symbol)
input ENUM_TIMEFRAMES InpTimeframe     = PERIOD_H1;     // Timeframe to export
input string          InpOutSubfolder  = "FOREX_EXPORT"; // Common\Files\<subfolder>\
input string          InpFileName      = "";            // If empty => SYMBOL_TF_dataset.csv
input bool            InpWriteHeader   = true;          // write header if file new
input bool            InpBackfillOnStart = true;        // fill missing closed bars since last export
input int             InpMaxBackfillBars = 200000;      // safety cap

// Indicator periods (MT5 native)
input int             InpEMA20         = 20;
input int             InpEMA50         = 50;
input int             InpRSI14         = 14;
input int             InpATR14         = 14;
input int             InpADX14         = 14;

//+------------------------------------------------------------------+
datetime g_last_written_bar_time = 0;

//+------------------------------------------------------------------+
string TfToString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return IntegerToString((int)tf);
   }
}

double PipSize(string symbol)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(digits == 3 || digits == 5) return 10.0 * point; // 5-digit FX / 3-digit JPY
   return point;
}

string DefaultFileName(const string symbol, ENUM_TIMEFRAMES tf)
{
   return symbol + "_" + TfToString(tf) + "_dataset.csv";
}

string BuildRelPath(const string subfolder, const string filename)
{
   if(subfolder == "") return filename;
   return subfolder + "\\" + filename;
}

bool FileExistsCommon(const string relpath)
{
   // Check existence by trying open for READ
   int h = FileOpen(relpath, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return false;
   FileClose(h);
   return true;
}

int OpenForAppendCommon(const string relpath, bool write_header_if_new)
{
   bool exists = FileExistsCommon(relpath);

   int h = FileOpen(relpath, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      Print("FileOpen failed: ", relpath, " err=", GetLastError());
      return INVALID_HANDLE;
   }

   // if new & header requested -> write header now (file pointer at 0)
   if(!exists && write_header_if_new)
   {
      FileWrite(h,
         "time","open","high","low","close","volume",
         "EMA_20","EMA_50","RSI_14","ATR_14_PIPS",
         "ADX_14","PLUS_DI_14","MINUS_DI_14"
      );
   }
   else
   {
      // move to end for append
      FileSeek(h, 0, SEEK_END);
   }
   return h;
}

// Read last exported time from last line (robust approach: scan last non-empty row)
datetime ReadLastTimeFromFileCommon(const string relpath)
{
   if(!FileExistsCommon(relpath))
      return 0;

   int h = FileOpen(relpath, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return 0;

   datetime last_t = 0;

   // Read through file; keep last parsed time
   while(!FileIsEnding(h))
   {
      string tstr = FileReadString(h); // time
      if(FileIsEnding(h)) break;

      // If header line, skip
      if(StringFind(tstr, "time") == 0)
      {
         // discard rest columns in header row
         for(int k=0;k<12;k++) { if(FileIsEnding(h)) break; FileReadString(h); }
         continue;
      }

      datetime t = (datetime)StringToTime(tstr);
      if(t > last_t) last_t = t;

      // discard remaining columns for this row (we already consumed 1)
      for(int k=0;k<12;k++) { if(FileIsEnding(h)) break; FileReadString(h); }
   }

   FileClose(h);
   return last_t;
}

bool AppendOneClosedBar(const string symbol, ENUM_TIMEFRAMES tf, int shift_closed, int file_handle,
                        int hEma20, int hEma50, int hRsi14, int hAtr14, int hAdx14)
{
   // shift_closed=1 means last closed bar
   datetime t = iTime(symbol, tf, shift_closed);
   if(t <= 0) return false;

   double o = iOpen(symbol, tf, shift_closed);
   double h = iHigh(symbol, tf, shift_closed);
   double l = iLow(symbol, tf, shift_closed);
   double c = iClose(symbol, tf, shift_closed);
   long   v = (long)iVolume(symbol, tf, shift_closed);

   // indicator values via CopyBuffer (1 value)
   double ema20[], ema50[], rsi[], atr[], adx[], pdi[], mdi[];
   ArraySetAsSeries(ema20, true);
   ArraySetAsSeries(ema50, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(adx, true);
   ArraySetAsSeries(pdi, true);
   ArraySetAsSeries(mdi, true);

   // CopyBuffer uses indicator series indexing by shift
   if(CopyBuffer(hEma20, 0, shift_closed, 1, ema20) != 1) return false;
   if(CopyBuffer(hEma50, 0, shift_closed, 1, ema50) != 1) return false;
   if(CopyBuffer(hRsi14, 0, shift_closed, 1, rsi)   != 1) return false;
   if(CopyBuffer(hAtr14, 0, shift_closed, 1, atr)   != 1) return false;

   // ADX buffers: 0=ADX, 1=+DI, 2=-DI
   if(CopyBuffer(hAdx14, 0, shift_closed, 1, adx) != 1) return false;
   if(CopyBuffer(hAdx14, 1, shift_closed, 1, pdi) != 1) return false;
   if(CopyBuffer(hAdx14, 2, shift_closed, 1, mdi) != 1) return false;

   double pip = PipSize(symbol);
   if(pip <= 0.0) pip = SymbolInfoDouble(symbol, SYMBOL_POINT);

   double atr_pips = atr[0] / pip;

   string tstr = TimeToString(t, TIME_DATE | TIME_MINUTES);

   FileWrite(file_handle,
      tstr,
      DoubleToString(o, 5),
      DoubleToString(h, 5),
      DoubleToString(l, 5),
      DoubleToString(c, 5),
      (string)v,
      DoubleToString(ema20[0], 5),
      DoubleToString(ema50[0], 5),
      DoubleToString(rsi[0], 6),
      DoubleToString(atr_pips, 6),
      DoubleToString(adx[0], 6),
      DoubleToString(pdi[0], 6),
      DoubleToString(mdi[0], 6)
   );

   return true;
}

void BackfillMissingBars(const string symbol, ENUM_TIMEFRAMES tf, int file_handle,
                         datetime last_written,
                         int hEma20, int hEma50, int hRsi14, int hAtr14, int hAdx14)
{
   // We want to append any closed bars with time > last_written up to shift=1
   int bars = Bars(symbol, tf);
   if(bars <= 2) return;

   // shift grows into the past. We'll find the first bar older/equal last_written, then export newer.
   // We'll scan from oldest to newest efficiently using iBarShift.
   int start_shift = -1;
   if(last_written > 0)
   {
      // shift of bar with time == last_written (or nearest)
      start_shift = iBarShift(symbol, tf, last_written, true);
   }

   // If not found, export up to MaxBackfillBars from the past? We choose last N bars conservatively.
   // But usually last_written will be found.
   int max_export = InpMaxBackfillBars;
   int exported = 0;

   // We export in chronological order: from older to newer.
   // Determine the oldest shift we need to export:
   // If last_written exists at shift S, then export shifts S-1 down to 1 (newer bars).
   int from_shift = 1;
   int to_shift = 1;

   if(start_shift > 1)
   {
      // last_written is in history; export bars newer than it: shifts (start_shift-1 .. 1)
      from_shift = start_shift - 1;
      to_shift = 1;
   }
   else if(last_written == 0)
   {
      // no prior file: export everything available but capped
      from_shift = MathMin(bars - 1, max_export);
      to_shift = 1;
   }
   else
   {
      // last_written set but not found: export recent max_export bars
      from_shift = MathMin(bars - 1, max_export);
      to_shift = 1;
   }

   for(int sh = from_shift; sh >= to_shift; sh--)
   {
      datetime t = iTime(symbol, tf, sh);
      if(last_written > 0 && t <= last_written) continue;

      if(AppendOneClosedBar(symbol, tf, sh, file_handle, hEma20, hEma50, hRsi14, hAtr14, hAdx14))
      {
         exported++;
         if(exported >= max_export) break;
      }
   }

   if(exported > 0)
      Print("Backfill exported bars: ", exported);
}

//+------------------------------------------------------------------+
// EA events
//+------------------------------------------------------------------+
int OnInit()
{
   string symbol = (InpSymbol == "" ? _Symbol : InpSymbol);
   if(!SymbolSelect(symbol, true))
   {
      Print("SymbolSelect failed for ", symbol);
      return INIT_FAILED;
   }

   // Build output path
   string filename = InpFileName;
   if(filename == "")
      filename = DefaultFileName(symbol, InpTimeframe);

   string relpath = BuildRelPath(InpOutSubfolder, filename);

   // Create indicator handles
   int hEma20 = iMA(symbol, InpTimeframe, InpEMA20, 0, MODE_EMA, PRICE_CLOSE);
   int hEma50 = iMA(symbol, InpTimeframe, InpEMA50, 0, MODE_EMA, PRICE_CLOSE);
   int hRsi14 = iRSI(symbol, InpTimeframe, InpRSI14, PRICE_CLOSE);
   int hAtr14 = iATR(symbol, InpTimeframe, InpATR14);
   int hAdx14 = iADX(symbol, InpTimeframe, InpADX14);

   if(hEma20==INVALID_HANDLE || hEma50==INVALID_HANDLE || hRsi14==INVALID_HANDLE || hAtr14==INVALID_HANDLE || hAdx14==INVALID_HANDLE)
   {
      Print("Indicator handle creation failed. err=", GetLastError());
      return INIT_FAILED;
   }

   // Persist handles in global variables via terminal globals? Not needed; we’ll store in static variables in OnTick.
   // Instead, we’ll attach them to chart via Global Variables? We can keep them static in file scope by using globals.
   // We'll just store them in static variables inside OnTick by using global variables:
   GlobalVariableSet("EXP_hEma20", hEma20);
   GlobalVariableSet("EXP_hEma50", hEma50);
   GlobalVariableSet("EXP_hRsi14", hRsi14);
   GlobalVariableSet("EXP_hAtr14", hAtr14);
   GlobalVariableSet("EXP_hAdx14", hAdx14);

   // Determine last written time from file
   g_last_written_bar_time = ReadLastTimeFromFileCommon(relpath);
   if(g_last_written_bar_time > 0)
      Print("Last exported time found in file: ", TimeToString(g_last_written_bar_time, TIME_DATE|TIME_MINUTES));
   else
      Print("No previous export found. Will create/append: ", relpath);

   // Open for append
   int fh = OpenForAppendCommon(relpath, InpWriteHeader);
   if(fh == INVALID_HANDLE)
   {
      // release handles
      IndicatorRelease(hEma20); IndicatorRelease(hEma50); IndicatorRelease(hRsi14); IndicatorRelease(hAtr14); IndicatorRelease(hAdx14);
      return INIT_FAILED;
   }

   // Optionally backfill on start
   if(InpBackfillOnStart)
   {
      BackfillMissingBars(symbol, InpTimeframe, fh, g_last_written_bar_time, hEma20, hEma50, hRsi14, hAtr14, hAdx14);

      // Update last written time after backfill (use latest closed bar time <= shift 1)
      datetime t1 = iTime(symbol, InpTimeframe, 1);
      if(t1 > g_last_written_bar_time)
         g_last_written_bar_time = t1;
   }

   FileClose(fh);

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   // Release indicator handles if stored
   int hEma20 = (int)GlobalVariableGet("EXP_hEma20");
   int hEma50 = (int)GlobalVariableGet("EXP_hEma50");
   int hRsi14 = (int)GlobalVariableGet("EXP_hRsi14");
   int hAtr14 = (int)GlobalVariableGet("EXP_hAtr14");
   int hAdx14 = (int)GlobalVariableGet("EXP_hAdx14");

   if(hEma20 > 0) IndicatorRelease(hEma20);
   if(hEma50 > 0) IndicatorRelease(hEma50);
   if(hRsi14 > 0) IndicatorRelease(hRsi14);
   if(hAtr14 > 0) IndicatorRelease(hAtr14);
   if(hAdx14 > 0) IndicatorRelease(hAdx14);

   GlobalVariableDel("EXP_hEma20");
   GlobalVariableDel("EXP_hEma50");
   GlobalVariableDel("EXP_hRsi14");
   GlobalVariableDel("EXP_hAtr14");
   GlobalVariableDel("EXP_hAdx14");
}

void OnTick()
{
   string symbol = (InpSymbol == "" ? _Symbol : InpSymbol);

   datetime closed_time = iTime(symbol, InpTimeframe, 1);
   if(closed_time <= 0) return;

   // Only on new closed candle
   if(closed_time <= g_last_written_bar_time)
      return;

   string filename = InpFileName;
   if(filename == "")
      filename = DefaultFileName(symbol, InpTimeframe);
   string relpath = BuildRelPath(InpOutSubfolder, filename);

   int fh = OpenForAppendCommon(relpath, InpWriteHeader);
   if(fh == INVALID_HANDLE) return;

   int hEma20 = (int)GlobalVariableGet("EXP_hEma20");
   int hEma50 = (int)GlobalVariableGet("EXP_hEma50");
   int hRsi14 = (int)GlobalVariableGet("EXP_hRsi14");
   int hAtr14 = (int)GlobalVariableGet("EXP_hAtr14");
   int hAdx14 = (int)GlobalVariableGet("EXP_hAdx14");

   // Backfill any missing bars between last written and now (e.g. terminal offline)
   if(InpBackfillOnStart)
      BackfillMissingBars(symbol, InpTimeframe, fh, g_last_written_bar_time, hEma20, hEma50, hRsi14, hAtr14, hAdx14);
   else
   {
      // Just append last closed bar
      AppendOneClosedBar(symbol, InpTimeframe, 1, fh, hEma20, hEma50, hRsi14, hAtr14, hAdx14);
   }

   FileClose(fh);

   g_last_written_bar_time = closed_time;

   Print("CSV appended. Last closed bar time: ", TimeToString(g_last_written_bar_time, TIME_DATE|TIME_MINUTES),
         " -> Common\\Files\\", (InpOutSubfolder=="" ? "" : (InpOutSubfolder + "\\")), filename);
}
