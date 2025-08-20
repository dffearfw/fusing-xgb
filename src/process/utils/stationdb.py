"""
Create on 2025/8/12

@auther:Thinkpad
"""
import os
import sqlite3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from geoalchemy2 import Geometry


class StationDatabase:
    def __init__(self, db_path="E:/data/gisws/stations.db"):
        # 解析路径并确保目录存在
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查权限
        if not os.access(str(self.db_path.parent), os.W_OK):
            raise PermissionError(f"没有写入权限: {self.db_path.parent}")

        # 尝试连接
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            print(f"成功连接到数据库: {self.db_path}")
            self._init_db()
        except sqlite3.OperationalError as e:
            # 更详细的错误处理
            if "unable to open database file" in str(e):
                if not self.db_path.parent.exists():
                    print(f"错误: 目录不存在 - {self.db_path.parent}")
                elif not self.db_path.parent.is_dir():
                    print(f"错误: 路径不是目录 - {self.db_path.parent}")
                else:
                    print(f"错误: 无法打开数据库文件 - 请检查文件权限")
            raise  # 重新抛出异常

    def _init_db(self):
        """初始化数据库"""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            station_ID TEXT PRIMARY KEY,
            Longitude REAL NOT NULL,
            Latitude REAL NOT NULL,
            Altitude REAL,
            Time TEXT,
            SWE REAL
            
        )
        """)

        self.conn.commit()

    def import_from_csv(self, csv_path):
        """从CSV导入数据"""
        df = pd.read_excel(csv_path)

        # 保存到数据库
        df.to_sql("stations", self.conn, if_exists="replace", index=False)

        self.conn.commit()
        return len(df)

    def get_stations(self, bbox=None, region=None):
        """查询站点"""
        query = "SELECT station_ID, Longitude, Latitude, SWE ,Time FROM stations"
        params = []

        if bbox:
            minx, miny, maxx, maxy = bbox
            query += " WHERE MBRContains(BuildMBR(?, ?, ?, ?), geometry)"
            params.extend([minx, miny, maxx, maxy])
        elif region:
            query += " WHERE region = ?"
            params.append(region)

        df = pd.read_sql_query(query, self.conn, params=params)
        return df.to_dict('records')

    def get_station(self,):
        """获取单个站点信息"""
        query = "SELECT * FROM stations "
        df = pd.read_sql_query(query, self.conn)
        return df.iloc[0].to_dict() if not df.empty else None

    def close(self):
        self.conn.close()


# 初始化数据库（只需运行一次）
if __name__ == "__main__":
    db = StationDatabase()
    count = db.import_from_csv("D:/pyworkspace/fusing xgb/config/sources/samples.xlsx")
    print(f"成功导入 {count} 个记录")

    print(db.get_station())
    db.close()
