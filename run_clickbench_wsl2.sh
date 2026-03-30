#!/bin/bash
# =============================================================================
# VelesDB vs ClickHouse — ClickBench Adapted Benchmark (WSL2)
# =============================================================================
set -euo pipefail
trap 'pkill -f "clickhouse-server" 2>/dev/null || true' EXIT

CH_BIN="$HOME/clickhouse-bench/clickhouse"
VENV_DIR="/tmp/bench-venv"
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"
PARQUET="/tmp/hits_1m.parquet"

# 0. Check parquet exists
if [ ! -f "$PARQUET" ]; then
    echo "=== Downloading ClickBench 1M rows ==="
    "$CH_BIN" local --query "
        SELECT * FROM url('https://datasets.clickhouse.com/hits_compatible/hits.parquet', Parquet)
        LIMIT 1000000
        INTO OUTFILE '$PARQUET'
        FORMAT Parquet
    "
    echo "  Downloaded: $(ls -lh $PARQUET | awk '{print $5}')"
fi

# 1. Start ClickHouse server
echo "=== Starting ClickHouse server ==="
mkdir -p "$HOME/ch-bench-data/tmp" "$HOME/ch-bench-data/log"
pkill -f "clickhouse-server" 2>/dev/null || true
sleep 1

"$CH_BIN" server \
    -L "$HOME/ch-bench-data/log/clickhouse.log" \
    -E "$HOME/ch-bench-data/log/clickhouse-err.log" \
    -- --path="$HOME/ch-bench-data/" --http_port=8123 --tcp_port=9000 --listen_host=127.0.0.1 --user_files_path=/tmp/ --tmp_path="$HOME/ch-bench-data/tmp/" &
CH_PID=$!

echo "  Waiting for server..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8123/ping >/dev/null 2>&1; then
        echo "  Server ready! (PID $CH_PID)"
        break
    fi
    sleep 1
done

# 2. Pre-import data into ClickHouse via native client (much faster than Python)
echo ""
echo "=== Importing ClickBench data into ClickHouse ==="
"$CH_BIN" client --host 127.0.0.1 --port 9000 --query "DROP TABLE IF EXISTS hits" 2>/dev/null || true
"$CH_BIN" client --host 127.0.0.1 --port 9000 --query "
    CREATE TABLE hits (
        WatchID UInt64, JavaEnable UInt8, Title String, GoodEvent Int16,
        EventTime UInt32, EventDate UInt16, CounterID UInt32,
        ClientIP UInt32, RegionID UInt32, UserID UInt64,
        CounterClass Int8, OS UInt8, UserAgent UInt8,
        URL String, Referer String, IsRefresh UInt8,
        RefererCategoryID UInt16, RefererRegionID UInt32,
        URLCategoryID UInt16, URLRegionID UInt32,
        ResolutionWidth UInt16, ResolutionHeight UInt16,
        ResolutionDepth UInt8, FlashMajor UInt8, FlashMinor UInt8,
        FlashMinor2 String, NetMajor UInt8, NetMinor UInt8,
        UserAgentMajor UInt16, UserAgentMinor String,
        CookieEnable UInt8, JavascriptEnable UInt8,
        IsMobile UInt8, MobilePhone UInt8, MobilePhoneModel String,
        Params String, IPNetworkID UInt32, TraficSourceID Int8,
        SearchEngineID UInt16, SearchPhrase String,
        AdvEngineID UInt8, IsArtificial UInt8, WindowClientWidth UInt16,
        WindowClientHeight UInt16, ClientTimeZone Int16,
        ClientEventTime UInt32, SilverlightVersion1 UInt8,
        SilverlightVersion2 UInt8, SilverlightVersion3 UInt32,
        SilverlightVersion4 UInt16, PageCharset String,
        CodeVersion UInt32, IsLink UInt8, IsDownload UInt8,
        IsNotBounce UInt8, FUniqID UInt64, OriginalURL String,
        HID UInt32, IsOldCounter UInt8, IsEvent UInt8,
        IsParameter UInt8, DontCountHits UInt8, WithHash UInt8,
        HitColor String, LocalEventTime UInt32, Age UInt8,
        Sex UInt8, Income UInt8, Interests UInt16,
        Robotness UInt8, RemoteIP UInt32, WindowName Int32,
        OpenerName Int32, HistoryLength Int16, BrowserLanguage String,
        BrowserCountry String, SocialNetwork String,
        SocialAction String, HTTPError UInt16, SendTiming UInt32,
        DNSTiming UInt32, ConnectTiming UInt32,
        ResponseStartTiming UInt32, ResponseEndTiming UInt32,
        FetchTiming UInt32, SocialSourceNetworkID UInt8,
        SocialSourcePage String, ParamPrice Int64,
        ParamOrderID String, ParamCurrency String,
        ParamCurrencyID UInt16, OpenstatServiceName String,
        OpenstatCampaignID String, OpenstatAdID String,
        OpenstatSourceID String, UTMSource String,
        UTMMedium String, UTMCampaign String, UTMContent String,
        UTMTerm String, FromTag String, HasGCLID UInt8,
        RefererHash UInt64, URLHash UInt64, CLID UInt32
    ) ENGINE = MergeTree()
    ORDER BY (CounterID, EventDate, UserID, EventTime, WatchID)
"
"$CH_BIN" client --host 127.0.0.1 --port 9000 --query "INSERT INTO hits SELECT * FROM file('$PARQUET', Parquet)"
ROW_COUNT=$("$CH_BIN" client --host 127.0.0.1 --port 9000 --query "SELECT count() FROM hits")
echo "  Loaded $ROW_COUNT rows into ClickHouse"

# 3. Run benchmark (skip CH import in Python, data already loaded)
echo ""
echo "=== Running ClickBench Benchmark ==="
source "$VENV_DIR/bin/activate"
python3 "$BENCH_DIR/bench_clickbench.py" --parquet "$PARQUET" --skip-ch-import "$@"

# 4. Cleanup (handled by trap)
echo ""
echo "  Done."
