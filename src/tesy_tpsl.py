"""
Test TP/SL integration
"""

from predictor import TradingPredictor
from trading_executor import create_executor_from_prediction
import pandas as pd
import numpy as np

def test_tpsl():
    # 1. Carica modello
    predictor = TradingPredictor()
    
    # Trova ultimo modello
    import glob, os
    model_files = glob.glob("models/*.h5")
    if not model_files:
        print("Nessun modello trovato")
        return
    
    model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    scaler_path = model_path.replace('.h5', '_scaler.pkl')
    
    predictor.load_model(model_path, scaler_path)
    
    # 2. Crea dati di esempio
    np.random.seed(42)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-12-01', periods=20, freq='H'),
        'price': 1.08 + np.random.randn(20).cumsum() * 0.001,
        'BB_UPPER': 1.082 + np.random.randn(20) * 0.0005,
        'BB_MIDDLE': 1.079 + np.random.randn(20) * 0.0005,
        'BB_LOWER': 1.076 + np.random.randn(20) * 0.0005,
        'MACD_MAIN': np.random.randn(20) * 0.0002,
        'MACD_SIGNAL': np.random.randn(20) * 0.00015,
        'MOMENTUM': np.random.randn(20) * 10 + 50,
        'RSI': np.random.uniform(30, 70, 20),
        'BULLPOWER': np.random.randn(20) * 0.0001,
        'BEARPOWER': np.random.randn(20) * -0.0001
    })
    
    # 3. Predici
    print("Effettuando predizione...")
    result = predictor.predict_single(test_data, return_confidence=True)
    
    # 4. Genera ordine
    print("\nGenerando ordine con TP/SL...")
    order_result = create_executor_from_prediction(result)
    
    if order_result['success']:
        print("\n" + "=" * 60)
        print("✅ ORDINE GENERATO!")
        print("=" * 60)
        print(order_result['display_text'])
        
        # Salva
        import json
        with open('test_order.json', 'w') as f:
            json.dump(order_result, f, indent=2)
        print("\nOrdine salvato in: test_order.json")
        
        # Mostra summary
        summary = order_result['execution_summary']
        print("\n📋 EXECUTION SUMMARY:")
        print(f"  Action: {summary['action']}")
        print(f"  Entry: {summary['entry']}")
        print(f"  TP: {summary['tp']} ({summary['reward_pips']} pips)")
        print(f"  SL: {summary['sl']} ({summary['risk_pips']} pips)")
        print(f"  R/R: {summary['rr_ratio']}:1")
        print(f"  Lots: {summary['position_size_lots']}")
        print(f"  Confidence: {summary['confidence']:.2f}")
    else:
        print(f"❌ Errore: {order_result.get('errors', ['Unknown error'])}")

if __name__ == "__main__":
    test_tpsl()