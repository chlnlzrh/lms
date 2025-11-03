#!/usr/bin/env python3
"""
Test Runner for Parallel Module Landing Page Testing
Handles server startup and comprehensive testing execution
"""

import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.server_process = None
        self.base_url = "http://localhost:3000"
        self.max_startup_wait = 60  # seconds
        
    def start_dev_server(self):
        """Start the Next.js development server"""
        print("Starting Next.js development server...")
        
        # Navigate to the Next.js app directory
        app_dir = Path(__file__).parent.parent / "data-engineering-platform"
        
        if not app_dir.exists():
            print(f"Application directory not found: {app_dir}")
            return False
            
        try:
            # Start the development server
            self.server_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=app_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"Server starting with PID: {self.server_process.pid}")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            return False
    
    def wait_for_server(self):
        """Wait for server to be ready"""
        print(f"Waiting for server at {self.base_url}...")
        
        start_time = time.time()
        while time.time() - start_time < self.max_startup_wait:
            try:
                response = requests.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"Server is ready! (took {time.time() - start_time:.1f}s)")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"Server failed to start within {self.max_startup_wait} seconds")
        return False
    
    def run_parallel_tests(self):
        """Execute the parallel module tests"""
        print("\nStarting parallel module landing page tests...")
        
        try:
            # Run the parallel test script
            result = subprocess.run([
                sys.executable, 
                "parallel_module_landing_test.py",
                self.base_url
            ], 
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Failed to run tests: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the development server"""
        if self.server_process:
            print("\n🛑 Stopping development server...")
            try:
                # Try graceful shutdown first
                self.server_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.server_process.kill()
                    self.server_process.wait()
                
                print("✅ Server stopped successfully")
                
            except Exception as e:
                print(f"⚠️  Error stopping server: {str(e)}")
    
    def run_complete_test_suite(self):
        """Run the complete test suite with server management"""
        print("Enhanced Module Landing Page Test Suite")
        print("=" * 60)
        print("Machine: 10 core/20 thread/32GB RAM")
        print("Mode: Parallel testing with 20 workers")
        print("Scope: All 20 module landing pages")
        print("=" * 60)
        
        success = False
        
        try:
            # Step 1: Start server
            if not self.start_dev_server():
                return False
            
            # Step 2: Wait for server to be ready
            if not self.wait_for_server():
                return False
            
            # Step 3: Run parallel tests
            success = self.run_parallel_tests()
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            
        except Exception as e:
            print(f"\n💥 Unexpected error: {str(e)}")
            
        finally:
            # Always clean up
            self.stop_server()
        
        return success

def main():
    """Main execution"""
    runner = TestRunner()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal, cleaning up...")
        runner.stop_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the complete test suite
    success = runner.run_complete_test_suite()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed or encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    main()