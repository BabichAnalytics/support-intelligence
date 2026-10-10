import csv
import tempfile
import unittest
from pathlib import Path
from app import Classifier,connect,import_tickets,priority,snapshot,training,ROOT

class PlatformTests(unittest.TestCase):
 def setUp(self):self.model=Classifier(training(ROOT/'training.csv'))
 def test_learned_category(self):self.assertEqual(self.model.predict('password login account locked')[0],'Access')
 def test_unknown_review(self):self.assertEqual(self.model.predict('xyzzy frobnicator'),('Needs review',0.0))
 def test_priority(self):self.assertEqual(priority('Network outage affects all users'),'High');self.assertEqual(priority('Single device slow'),'Normal')
 def test_import_is_idempotent(self):
  with connect(':memory:') as db:
   import_tickets(db,ROOT/'sample_tickets.csv',self.model);import_tickets(db,ROOT/'sample_tickets.csv',self.model)
   self.assertEqual(len(snapshot(db)['tickets']),150)
 def test_invalid_batch_preserves_database(self):
  with tempfile.TemporaryDirectory() as folder,connect(':memory:') as db:
   path=Path(folder)/'bad.csv';path.write_text('id,created,subject,description,status\nA,2025-01-01,Test,Test,Open\nB,invalid,Test,Test,Open\n')
   with self.assertRaises(ValueError):import_tickets(db,path,self.model)
   self.assertEqual(len(snapshot(db)['tickets']),0)
 def test_evaluation(self):
  with connect(':memory:') as db:
   import_tickets(db,ROOT/'sample_tickets.csv',self.model)
   report=snapshot(db)['evaluation'];self.assertEqual(report['labeled_records'],150);self.assertGreaterEqual(report['accuracy'],0);self.assertLessEqual(report['accuracy'],1)
if __name__=='__main__':unittest.main()
