"""Support Intelligence: local ticket analytics and machine-learning classification."""
import argparse
from collections import Counter
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
CATEGORIES=['Access','Network','Hardware','Software','Billing']

def tokens(text):
    return re.findall(r'[a-z]{2,}',text.lower())

class Classifier:
    """Multinomial Naive Bayes trained locally; scores are not calibrated confidence."""
    def __init__(self, examples):
        self.counts={c:Counter() for c in CATEGORIES};self.docs=Counter();self.vocab=set()
        for text,label in examples:
            self.docs[label]+=1;words=tokens(text);self.counts[label].update(words);self.vocab.update(words)
        self.total=sum(self.docs.values())
        if not self.total: raise ValueError('Training data is empty.')
    def predict(self,text):
        words=[w for w in tokens(text) if w in self.vocab]
        if not words:return 'Needs review',0.0
        scores={}
        for c in CATEGORIES:
            score=math.log((self.docs[c]+1)/(self.total+len(CATEGORIES)))
            denom=sum(self.counts[c].values())+len(self.vocab)
            score+=sum(math.log((self.counts[c][w]+1)/denom) for w in words)
            scores[c]=score
        best=max(scores,key=scores.get);peak=scores[best]
        score=1/sum(math.exp(s-peak) for s in scores.values())
        return best,round(score,4)

def training(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Training data is empty.')
    if any(r.get('category') not in CATEGORIES or not r.get('text','').strip() for r in rows):raise ValueError('Invalid training examples.')
    return [(r['text'],r['category']) for r in rows]

def priority(text):
    # Explicit operational rules, independent of the ML category model.
    return 'High' if any(w in text.lower() for w in ['all users','outage','security breach','production down']) else 'Normal'

def connect(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, created TEXT NOT NULL, subject TEXT NOT NULL, description TEXT NOT NULL, status TEXT NOT NULL, category TEXT NOT NULL, model_score REAL NOT NULL, priority TEXT NOT NULL, expected_category TEXT)')
    return db

def import_tickets(db,path,model):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        required={'id','created','subject','description','status'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Required headers: id,created,subject,description,status')
        validated=[];seen=set()
        for line,r in enumerate(reader,2):
            if not all(r.get(k,'').strip() for k in required):raise ValueError(f'Record {line}: blank required field')
            datetime.strptime(r['created'],'%Y-%m-%d')
            if r['id'] in seen:raise ValueError(f'Record {line}: duplicate ticket ID')
            seen.add(r['id'])
            if r['status'] not in ['Open','In progress','Resolved']:raise ValueError(f'Record {line}: invalid status')
            expected=r.get('expected_category') or None
            if expected and expected not in CATEGORIES:raise ValueError(f'Record {line}: invalid expected category')
            category,score=model.predict(r['subject']+' '+r['description'])
            validated.append((r['id'],r['created'],r['subject'],r['description'],r['status'],category,score,priority(r['subject']+' '+r['description']),expected))
    with db:
        db.executemany('INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET created=excluded.created,subject=excluded.subject,description=excluded.description,status=excluded.status,category=excluded.category,model_score=excluded.model_score,priority=excluded.priority,expected_category=excluded.expected_category',validated)
    return len(validated)

def snapshot(db):
    rows=[dict(r) for r in db.execute('SELECT * FROM tickets ORDER BY created DESC,id')]
    labeled=[r for r in rows if r['expected_category']]
    accuracy=sum(r['category']==r['expected_category'] for r in labeled)/len(labeled) if labeled else None
    return {'tickets':rows,'evaluation':{'labeled_records':len(labeled),'accuracy':accuracy,'note':'Synthetic demonstration evaluation; not evidence of real-world accuracy.'},'model':'Local Naive Bayes; relative model scores are not calibrated confidence.'}

TEMPLATES={
'Access':'Please confirm the exact sign-in error and whether other approved accounts can sign in. Use the approved password-reset process; never send passwords.',
'Network':'Please confirm which devices and locations are affected and when connectivity last worked. Capture the error message for the network team.',
'Hardware':'Please share the device model, symptoms, and any indicator lights. Follow your approved hardware troubleshooting procedure.',
'Software':'Please share the application version, exact error, and steps that reproduce the issue. Note any recent changes.',
'Billing':'Please share the invoice reference and describe the discrepancy using your approved billing channel. Avoid sending full payment-card details.',
'Needs review':'Please provide the exact symptoms and steps to reproduce the issue so a support specialist can review it.'}

def handler_for(db_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':body=(ROOT/'index.html').read_bytes();mime='text/html; charset=utf-8'
            elif path=='/api/tickets':
                with connect(db_path) as db:data=snapshot(db)
                body=json.dumps(data,allow_nan=False).encode();mime='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,default=ROOT/'sample_tickets.csv');p.add_argument('--db',type=Path,default=ROOT/'support.db');p.add_argument('--port',type=int,default=8002);p.add_argument('--import-only',action='store_true');args=p.parse_args()
    try:
        model=Classifier(training(ROOT/'training.csv'))
        with connect(args.db) as db:count=import_tickets(db,args.csv,model);report=snapshot(db)
        print(f'Imported or updated {count} tickets. Evaluation: {report["evaluation"]}',flush=True)
        if args.import_only:return
        server=HTTPServer(('127.0.0.1',args.port),handler_for(args.db))
    except (OSError,ValueError,csv.Error,sqlite3.Error) as e:p.exit(1,f'Error: {e}\n')
    print(f'Open http://127.0.0.1:{args.port} — Ctrl+C to stop',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
