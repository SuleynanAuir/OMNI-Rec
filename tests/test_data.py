import unittest
import tempfile
import os
import pandas as pd
import json
import random
from minionerec.data import (
    SFTData, D3Dataset, EvalD3Dataset, EvalSidDataset,
    SidDataset, SidSFTDataset, SidItemFeatDataset, RLTitle2SidDataset,
    RLSid2TitleDataset, RLSidhis2TitleDataset, FusionSeqRecDataset,
    TitleHistory2SidSFTDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset
)
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.bos_token_id = 2
    
    def encode(self, text, bos=False, eos=False):
        # Simple mock encoding - just return list of integers based on text length
        tokens = list(range(10, 10 + min(len(text), 50)))  # Limit to 50 tokens max
        if bos:
            tokens = [self.bos_token_id] + tokens
        if eos:
            tokens = tokens + [self.eos_token_id]
        return tokens

def create_minimal_csv(file_path, data):
    """Helper to create minimal CSV files for testing"""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

class TestDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MockTokenizer()
        
        # Create temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_dir.name
        
        # Create sample CSV data
        cls.csv_data = {
            'history_item_title': ["['Item A', 'Item B']", "['Item C', 'Item D']"],
            'item_title': ['Item E', 'Item F'],
            'history_item_id': ["['1', '2']", "['3', '4']"],
            'item_id': ['5', '6'],
            'history_item_sid': ["['SID1', 'SID2']", "['SID3', 'SID4']"],
            'item_sid': ['SID5', 'SID6'],
            'user_id_original_str': ['user1', 'user2'],
            'e_token': ['[CTX_HOMEPAGE]', '[CTX_SEARCH]']
        }
        cls.csv_file = os.path.join(cls.temp_path, 'test_data.csv')
        create_minimal_csv(cls.csv_file, cls.csv_data)
        
        # Create sample item features JSON
        cls.item_features = {
            '5': {'title': 'Item E', 'description': 'Description of Item E', 'item_type': 'O'},
            '6': {'title': 'Item F', 'description': 'Description of Item F', 'item_type': 'I'}
        }
        cls.item_file = os.path.join(cls.temp_path, 'test.item.json')
        with open(cls.item_file, 'w') as f:
            json.dump(cls.item_features, f)
        
        # Create sample indices JSON
        cls.indices = {
            '5': ['SID5_1', 'SID5_2', 'SID5_3'],
            '6': ['SID6_1', 'SID6_2', 'SID6_3']
        }
        cls.index_file = os.path.join(cls.temp_path, 'test.index.json')
        with open(cls.index_file, 'w') as f:
            json.dump(cls.indices, f)
        
        # Create sample user preference JSON
        cls.user_preference_data = [
            {
                'user': 'user1',
                'user_preference': 'Likes action games',
                'context': {
                    'history_items': ['1', '2'],
                    'target_item': '5'
                },
                'split': 'train'
            },
            {
                'user': 'user2',
                'user_preference': 'Prefers strategy games',
                'context': {
                    'history_items': ['3', '4'],
                    'target_item': '6'
                },
                'split': 'train'
            }
        ]
        cls.preference_file = os.path.join(cls.temp_path, 'test_preference.json')
        with open(cls.preference_file, 'w') as f:
            json.dump(cls.user_preference_data, f)

    @classmethod
    def tearDownClass(cls):
        # Cleanup temporary directory
        cls.temp_dir.cleanup()

    def test_SFTData_initialization(self):
        """Test SFTData initialization"""
        dataset = SFTData(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_D3Dataset_initialization(self):
        """Test D3Dataset initialization"""
        dataset = D3Dataset(
            train_file=self.csv_file,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_EvalD3Dataset_initialization(self):
        """Test EvalD3Dataset initialization"""
        dataset = EvalD3Dataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidDataset_initialization(self):
        """Test SidDataset initialization"""
        dataset = SidDataset(
            train_file=self.csv_file,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidSFTDataset_initialization(self):
        """Test SidSFTDataset initialization"""
        dataset = SidSFTDataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SFTData_initialization(self):
        """Test SFTData initialization"""
        dataset = SFTData(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidItemFeatDataset_initialization(self):
        """Test SidItemFeatDataset initialization"""
        dataset = SidItemFeatDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 2)  # Should have at least 2 samples (sid2title and title2sid)
        self.assertTrue(hasattr(dataset, 'inputs'))
        
    def test_EvalSidDataset_initialization(self):
        """Test SidItemFeatDataset initialization"""
        dataset = EvalSidDataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )

        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLTitle2SidDataset_initialization(self):
        """Test RLTitle2SidDataset initialization"""
        dataset = RLTitle2SidDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 2)  # Should have at least 2 samples
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLSid2TitleDataset_initialization(self):
        """Test RLSid2TitleDataset initialization"""
        dataset = RLSid2TitleDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 1)  # Should have at least 1 sample
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLSidhis2TitleDataset_initialization(self):
        """Test RLSidhis2TitleDataset initialization"""
        dataset = RLSidhis2TitleDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_FusionSeqRecDataset_initialization(self):
        """Test FusionSeqRecDataset initialization"""
        dataset = FusionSeqRecDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_TitleHistory2SidSFTDataset_initialization(self):
        """Test TitleHistory2SidSFTDataset initialization"""
        dataset = TitleHistory2SidSFTDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_PreferenceSFTDataset_initialization(self):
        """Test PreferenceSFTDataset initialization"""
        dataset = PreferenceSFTDataset(
            user_preference_file=self.preference_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_UserPreference2sidSFTDataset_initialization(self):
        """Test UserPreference2sidSFTDataset initialization"""
        dataset = UserPreference2sidSFTDataset(
            user_preference_file=self.preference_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

if __name__ == '__main__':
    # Run the tests
    unittest.main()
