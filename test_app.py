import unittest
import json
from App import app, load_stock_data, build_model, stock_mapping

class StockAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_load_stock_data(self):
        X, y, stock_ids = load_stock_data()
        self.assertTrue(len(X) > 0, "X should not be empty")
        self.assertTrue(len(y) > 0, "y should not be empty")
        self.assertEqual(X.shape[0], y.shape[0], "X and y should have the same number of samples")
        self.assertEqual(len(stock_ids), y.shape[0], "stock_ids should match number of y samples")

    def test_model_prediction_shape(self):
        X, y, stock_ids = load_stock_data()
        model = build_model()
        preds = model.predict([X[:10], stock_ids[:10]])
        self.assertEqual(preds.shape, (10, 1), "Prediction shape should be (10, 1)")

    def test_rankings_endpoint(self):
        response = self.app.get('/rankings')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
        self.assertTrue(len(data) > 0, "Should return stock rankings")
        for stock, info in data.items():
            self.assertIn("investment score", info[0], "Missing 'investment score'")
            self.assertIn("predicted price", info[0], "Missing 'predicted price'")

if __name__ == '__main__':
    unittest.main()
