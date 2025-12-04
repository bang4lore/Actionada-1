# Классификация банковских транзакций

## EDA выводы
- **Категории**: 15 классов, дисбаланс (топ-3 = 60%)
- **Пропуски**: Merchant — 2%, заполнены 'UNKNOWN'
- **Transaction_Amount**: дебет < кредит, правый хвост
- **Merchant**: топ-10 = 40% транзакций

## Как запустить
pip install -r requirements.txt
python banking_pipeline.py

## Результат
Macro F1: 0.82

## Предобработка
- Merchant > CountVectorizer
- Date > день недели, месяц, выходной
- Сумма > StandardScaler
- Тип > OneHotEncoder