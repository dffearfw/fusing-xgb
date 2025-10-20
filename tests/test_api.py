import pytest
from process.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """测试健康检查端点"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_status_endpoint(client):
    """测试状态端点"""
    response = client.get('/api/status')
    assert response.status_code == 200
    data = response.get_json()
    assert 'download' in data
    assert 'processing' in data
    assert 'training' in data