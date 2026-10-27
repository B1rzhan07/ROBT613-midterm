"""Regression for the HTTP 429 observed on the fresh Colab runtime."""
import unittest,io
from unittest.mock import patch
import requests
from remotezip import RemoteFetcher
import fetch_training as download

class RetryTest(unittest.TestCase):
    def test_429_honors_server_delay_then_succeeds(self):
        response=requests.Response();response.raw=io.BytesIO();response.status_code=429;response.headers['Retry-After']='2'
        error=requests.HTTPError('rate limited',response=response)
        download._LAST_REQUEST=0
        with patch.object(RemoteFetcher,'_request',side_effect=[error,('bytes','bytes 0-9/10')]) as request, patch.object(download.time,'sleep') as sleep:
            result=download.ParallelRangeFetcher('https://zenodo.org/example')._request({})
        self.assertEqual(result,('bytes','bytes 0-9/10'))
        self.assertEqual(request.call_count,2)
        self.assertIn(unittest.mock.call(2.),sleep.call_args_list)

    def test_permanent_404_is_not_retried(self):
        response=requests.Response();response.status_code=404
        error=requests.HTTPError('missing',response=response)
        download._LAST_REQUEST=0
        with patch.object(RemoteFetcher,'_request',side_effect=error) as request, patch.object(download.time,'sleep'):
            with self.assertRaises(requests.HTTPError):
                download.ParallelRangeFetcher('https://zenodo.org/example')._request({})
        self.assertEqual(request.call_count,1)

if __name__=='__main__':unittest.main()
