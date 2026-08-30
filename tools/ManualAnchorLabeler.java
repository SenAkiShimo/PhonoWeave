import javax.sound.sampled.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.List;

public class ManualAnchorLabeler {
    static final class Item {
        int promptIndex;
        String baseUnit;
        String context;
        String syllable;
        String className;
        Path wav;
        int occurrence;
        double prevCueMs;
        double cueMs;
        double nextCueMs;
    }

    static final class Label {
        double anchorMsAfterCue;
        boolean uncertain;
        Label(double anchorMsAfterCue, boolean uncertain) {
            this.anchorMsAfterCue = anchorMsAfterCue;
            this.uncertain = uncertain;
        }
    }

    static final class AudioData {
        final float[] samples;
        final float sampleRate;
        final AudioFormat playbackFormat;
        final byte[] playbackBytes;
        AudioData(float[] samples, float sampleRate, AudioFormat playbackFormat, byte[] playbackBytes) {
            this.samples = samples;
            this.sampleRate = sampleRate;
            this.playbackFormat = playbackFormat;
            this.playbackBytes = playbackBytes;
        }
    }

    private final List<Item> items;
    private final Path labelsPath;
    private final Map<String, Label> labels = new HashMap<>();
    private final Map<Path, AudioData> audioCache = new HashMap<>();
    private int index = 0;

    private final JFrame frame = new JFrame("PhonoWeave Manual Labels");
    private final JLabel title = new JLabel();
    private final JLabel meta = new JLabel();
    private final JLabel progress = new JLabel();
    private final JLabel value = new JLabel(" ");
    private final WavePanel wave = new WavePanel();
    private final JButton prev = new JButton("←");
    private final JButton play = new JButton("Play  Space");
    private final JButton uncertain = new JButton("Uncertain  U");
    private final JButton clear = new JButton("Clear  Del");
    private final JButton next = new JButton("→");

    private volatile Clip activeClip;

    ManualAnchorLabeler(List<Item> items, Path labelsPath) throws IOException {
        this.items = items;
        this.labelsPath = labelsPath;
        loadLabels();
        buildUi();
        showItem();
    }

    private static String key(int promptIndex, int occurrence) {
        return promptIndex + ":" + occurrence;
    }

    private String currentKey() {
        Item item = items.get(index);
        return key(item.promptIndex, item.occurrence);
    }

    private void buildUi() {
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setMinimumSize(new Dimension(950, 520));
        frame.setSize(1180, 680);
        frame.setLocationByPlatform(true);

        title.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 28));
        meta.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        progress.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        value.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));

        JPanel head = new JPanel(new BorderLayout(8, 4));
        JPanel names = new JPanel();
        names.setLayout(new BoxLayout(names, BoxLayout.Y_AXIS));
        names.add(title);
        names.add(meta);
        head.add(names, BorderLayout.CENTER);
        head.add(progress, BorderLayout.EAST);

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        controls.add(prev);
        controls.add(play);
        controls.add(uncertain);
        controls.add(clear);
        controls.add(next);
        controls.add(Box.createHorizontalStrut(14));
        controls.add(value);

        JPanel root = new JPanel(new BorderLayout(0, 12));
        root.setBorder(BorderFactory.createEmptyBorder(16, 18, 16, 18));
        root.add(head, BorderLayout.NORTH);
        root.add(wave, BorderLayout.CENTER);
        root.add(controls, BorderLayout.SOUTH);
        frame.setContentPane(root);

        prev.addActionListener(e -> move(-1));
        next.addActionListener(e -> move(1));
        play.addActionListener(e -> playCurrent());
        uncertain.addActionListener(e -> toggleUncertain());
        clear.addActionListener(e -> clearCurrent());

        JRootPane rp = frame.getRootPane();
        bind(rp, "SPACE", "play", this::playCurrent);
        bind(rp, "LEFT", "prev", () -> move(-1));
        bind(rp, "RIGHT", "next", () -> move(1));
        bind(rp, "U", "uncertain", this::toggleUncertain);
        bind(rp, "DELETE", "clear", this::clearCurrent);
        bind(rp, "BACK_SPACE", "clear2", this::clearCurrent);

        wave.addMouseListener(new MouseAdapter() {
            @Override public void mousePressed(MouseEvent e) {
                markAtX(e.getX());
            }
        });

        frame.setVisible(true);
    }

    private static void bind(JRootPane rp, String stroke, String name, Runnable action) {
        KeyStroke key = KeyStroke.getKeyStroke(stroke);
        rp.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(key, name);
        rp.getActionMap().put(name, new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { action.run(); }
        });
    }

    private void showItem() {
        Item item = items.get(index);
        title.setText(item.syllable + "   " + item.occurrence + "/2");
        meta.setText(item.baseUnit + " · " + item.context + " · " + item.className);
        progress.setText((index + 1) + "/" + items.size() + "     saved " + labels.size() + "/" + items.size());
        Label label = labels.get(currentKey());
        if (label == null) {
            value.setText(" ");
            uncertain.setText("Uncertain  U");
        } else {
            value.setText(String.format(Locale.US, "%+.1f ms%s", label.anchorMsAfterCue, label.uncertain ? "  ?" : ""));
            uncertain.setText(label.uncertain ? "Certain  U" : "Uncertain  U");
        }
        prev.setEnabled(index > 0);
        next.setEnabled(index < items.size() - 1);
        loadAudio(item.wav);
        wave.repaint();
    }

    private void move(int delta) {
        int ni = Math.max(0, Math.min(items.size() - 1, index + delta));
        if (ni != index) {
            stopPlayback();
            index = ni;
            showItem();
        }
    }

    private void markAtX(int x) {
        if (wave.getWidth() <= 1) return;
        Item item = items.get(index);
        double start = item.prevCueMs;
        double end = item.nextCueMs;
        double frac = Math.max(0.0, Math.min(1.0, x / (double) wave.getWidth()));
        double absoluteMs = start + frac * (end - start);
        double relativeMs = absoluteMs - item.cueMs;
        labels.put(currentKey(), new Label(relativeMs, false));
        saveLabels();
        wave.repaint();
        if (index < items.size() - 1) {
            index++;
            showItem();
        } else {
            showItem();
        }
    }

    private void toggleUncertain() {
        Label label = labels.get(currentKey());
        if (label == null) return;
        label.uncertain = !label.uncertain;
        saveLabels();
        showItem();
    }

    private void clearCurrent() {
        labels.remove(currentKey());
        saveLabels();
        showItem();
    }

    private void stopPlayback() {
        Clip clip = activeClip;
        if (clip != null) {
            clip.stop();
            clip.close();
            activeClip = null;
        }
    }

    private void playCurrent() {
        stopPlayback();
        try {
            Item item = items.get(index);
            AudioData audio = loadAudio(item.wav);
            int frameSize = audio.playbackFormat.getFrameSize();
            float sr = audio.playbackFormat.getFrameRate();
            double startMs = Math.max(0.0, item.prevCueMs - 80.0);
            double endMs = Math.min(audio.samples.length * 1000.0 / audio.sampleRate, item.nextCueMs + 80.0);
            int startFrame = Math.max(0, (int)Math.round(startMs * sr / 1000.0));
            int endFrame = Math.min(audio.playbackBytes.length / frameSize, (int)Math.round(endMs * sr / 1000.0));
            int byteStart = startFrame * frameSize;
            int byteLen = Math.max(0, (endFrame - startFrame) * frameSize);
            if (byteLen <= 0) return;
            Clip clip = AudioSystem.getClip();
            clip.open(audio.playbackFormat, audio.playbackBytes, byteStart, byteLen);
            activeClip = clip;
            clip.addLineListener(ev -> {
                if (ev.getType() == LineEvent.Type.STOP && activeClip == clip) {
                    clip.close();
                    activeClip = null;
                }
            });
            clip.start();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(frame, ex.toString(), "Playback error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private AudioData loadAudio(Path path) {
        AudioData cached = audioCache.get(path);
        if (cached != null) return cached;
        try {
            AudioInputStream original = AudioSystem.getAudioInputStream(path.toFile());
            AudioFormat src = original.getFormat();
            AudioFormat pcm = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    src.getSampleRate(),
                    16,
                    src.getChannels(),
                    src.getChannels() * 2,
                    src.getSampleRate(),
                    false);
            AudioInputStream decoded = AudioSystem.getAudioInputStream(pcm, original);
            byte[] bytes = decoded.readAllBytes();
            int channels = pcm.getChannels();
            int frameSize = pcm.getFrameSize();
            int frames = bytes.length / frameSize;
            float[] mono = new float[frames];
            for (int i = 0; i < frames; i++) {
                double sum = 0.0;
                int off = i * frameSize;
                for (int ch = 0; ch < channels; ch++) {
                    int p = off + ch * 2;
                    int lo = bytes[p] & 0xff;
                    int hi = bytes[p + 1];
                    short sample = (short)((hi << 8) | lo);
                    sum += sample / 32768.0;
                }
                mono[i] = (float)(sum / channels);
            }
            AudioData result = new AudioData(mono, pcm.getSampleRate(), pcm, bytes);
            audioCache.put(path, result);
            return result;
        } catch (Exception ex) {
            throw new RuntimeException("Cannot read " + path + ": " + ex, ex);
        }
    }

    private void loadLabels() throws IOException {
        if (!Files.isRegularFile(labelsPath)) return;
        for (String line : Files.readAllLines(labelsPath, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.split("\\t", -1);
            if (p.length < 4) continue;
            try {
                int prompt = Integer.parseInt(p[0]);
                int occ = Integer.parseInt(p[1]);
                double ms = Double.parseDouble(p[2]);
                boolean un = "uncertain".equals(p[3]);
                labels.put(key(prompt, occ), new Label(ms, un));
            } catch (NumberFormatException ignored) {}
        }
    }

    private void saveLabels() {
        try {
            Files.createDirectories(labelsPath.getParent());
            List<String> out = new ArrayList<>();
            out.add("# prompt_index\toccurrence\tanchor_ms_after_cue\tstatus");
            for (Item item : items) {
                Label label = labels.get(key(item.promptIndex, item.occurrence));
                if (label != null) {
                    out.add(String.format(Locale.US, "%d\t%d\t%.3f\t%s",
                            item.promptIndex, item.occurrence, label.anchorMsAfterCue,
                            label.uncertain ? "uncertain" : "ok"));
                }
            }
            Files.write(labelsPath, out, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(frame, ex.toString(), "Save error", JOptionPane.ERROR_MESSAGE);
        }
    }

    final class WavePanel extends JPanel {
        WavePanel() {
            setPreferredSize(new Dimension(1000, 430));
            setBackground(Color.WHITE);
            setBorder(BorderFactory.createLineBorder(new Color(150, 150, 150)));
            setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        }

        @Override protected void paintComponent(Graphics raw) {
            super.paintComponent(raw);
            if (items.isEmpty()) return;
            Graphics2D g = (Graphics2D) raw.create();
            try {
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                Item item = items.get(index);
                AudioData audio = loadAudio(item.wav);
                int w = getWidth();
                int h = getHeight();
                double startMs = item.prevCueMs;
                double endMs = item.nextCueMs;
                int start = Math.max(0, (int)Math.floor(startMs * audio.sampleRate / 1000.0));
                int end = Math.min(audio.samples.length, (int)Math.ceil(endMs * audio.sampleRate / 1000.0));
                int span = Math.max(1, end - start);
                int mid = h / 2;

                g.setColor(new Color(228, 228, 228));
                g.drawLine(0, mid, w, mid);
                g.setColor(new Color(30, 30, 30));
                for (int x = 0; x < w; x++) {
                    int a = start + (int)((long)x * span / Math.max(1, w));
                    int b = start + (int)((long)(x + 1) * span / Math.max(1, w));
                    b = Math.max(a + 1, Math.min(end, b));
                    float peak = 0f;
                    for (int i = a; i < b; i++) peak = Math.max(peak, Math.abs(audio.samples[i]));
                    int amp = (int)(peak * h * 0.44);
                    g.drawLine(x, mid - amp, x, mid + amp);
                }

                drawCue(g, item.prevCueMs, startMs, endMs, w, h, new Color(145,145,145), "prev");
                drawCue(g, item.cueMs, startMs, endMs, w, h, new Color(34,91,151), "cue");
                drawCue(g, item.nextCueMs, startMs, endMs, w, h, new Color(115,79,150), "next");

                Label label = labels.get(currentKey());
                if (label != null) {
                    double absolute = item.cueMs + label.anchorMsAfterCue;
                    int x = (int)Math.round((absolute - startMs) / (endMs - startMs) * w);
                    g.setColor(new Color(176, 45, 45));
                    g.setStroke(new BasicStroke(3f));
                    g.drawLine(x, 0, x, h);
                }
            } finally {
                g.dispose();
            }
        }

        private void drawCue(Graphics2D g, double time, double start, double end, int w, int h, Color color, String name) {
            int x = (int)Math.round((time - start) / (end - start) * w);
            g.setColor(color);
            g.setStroke(new BasicStroke(2f));
            g.drawLine(x, 0, x, h);
            g.drawString(name, Math.min(w - 35, x + 4), 16);
        }
    }

    static List<Item> readManifest(Path path) throws IOException {
        List<Item> result = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.split("\\t", -1);
            if (p.length < 10) throw new IOException("Bad manifest row: " + line);
            Item item = new Item();
            item.promptIndex = Integer.parseInt(p[0]);
            item.baseUnit = p[1];
            item.context = p[2];
            item.syllable = p[3];
            item.className = p[4];
            item.wav = Paths.get(p[5]);
            item.occurrence = Integer.parseInt(p[6]);
            item.prevCueMs = Double.parseDouble(p[7]);
            item.cueMs = Double.parseDouble(p[8]);
            item.nextCueMs = Double.parseDouble(p[9]);
            result.add(item);
        }
        return result;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: java ManualAnchorLabeler.java MANIFEST.tsv LABELS.tsv");
            System.exit(2);
        }
        Path manifest = Paths.get(args[0]).toAbsolutePath();
        Path labels = Paths.get(args[1]).toAbsolutePath();
        List<Item> items = readManifest(manifest);
        SwingUtilities.invokeLater(() -> {
            try {
                new ManualAnchorLabeler(items, labels);
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(null, ex.toString(), "PhonoWeave", JOptionPane.ERROR_MESSAGE);
            }
        });
    }
}
