package water.hadoop;

import org.junit.*;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

public class EmbeddedH2OConfigTest {

  private ServerSocket _blocked_server_socket;
  private int _blocked_port;

  private ServerSocket _open_server_socket;
  private int _open_port;

  @Before
  public void beforeTest() throws IOException {
    _blocked_server_socket = new ServerSocket(0, 1);
    _blocked_port = _blocked_server_socket.getLocalPort();
    // fill backlog queue by this request so consequent requests will be blocked
    new Socket().connect(_blocked_server_socket.getLocalSocketAddress());

    _open_server_socket = new ServerSocket(0);
    _open_port = _open_server_socket.getLocalPort();
  }

  @Test
  public void testFetchFile_failure() throws Exception {
    ExCollectingEmbeddedH2OConfig cfg = new ExCollectingEmbeddedH2OConfig();
    cfg.setDriverCallbackIp("127.0.0.1");
    cfg.setDriverCallbackPort(_blocked_port);

    SocketException e = null;
    try {
      cfg.fetchFlatfile();
    } catch (SocketException se) {
      e = se;
    }
    assertNotNull(e);

    assertEquals(2, cfg.exceptions.size());
  }

  @Test
  public void testFetchFile() throws Exception {
    new h2odriver()
            .new CallbackManager(_open_server_socket, 1)
            .start();

    EmbeddedH2OConfig cfg = new SocketClosingEmbeddedH2OConfig();
    cfg.setDriverCallbackIp("127.0.0.1");
    cfg.setDriverCallbackPort(_blocked_port);
    cfg.setEmbeddedWebServerInfo("h2o.ai", 600);

    String flatfile = cfg.fetchFlatfile();
    assertEquals("h2o.ai:600\n", flatfile);
  }
  
  @After
  public void afterTest() throws IOException {
    if (_blocked_server_socket != null && !_blocked_server_socket.isClosed()) {
      _blocked_server_socket.close();
    }
  }
  
  private static class ExCollectingEmbeddedH2OConfig extends EmbeddedH2OConfig {
    private final List<IOException> exceptions = new LinkedList<>();
    @Override
    protected void reportFetchfileAttemptFailure(IOException ioex, int attempt) throws IOException {
      assertEquals(attempt, exceptions.size());
      exceptions.add(ioex);
      super.reportFetchfileAttemptFailure(ioex, attempt);
    }
  }

  private class SocketClosingEmbeddedH2OConfig extends EmbeddedH2OConfig {
    @Override
    protected void reportFetchfileAttemptFailure(IOException ioex, int attempt) throws IOException {
      setDriverCallbackPort(_open_port);
    }
  }

}
