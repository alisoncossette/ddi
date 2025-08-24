"""
Dynamic Data Intelligence - Data Lineage Tracking
Tracks data flow and dependencies across AI systems
"""

import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from enum import Enum

class DataAssetType(Enum):
    SOURCE = "source"
    TRANSFORMATION = "transformation"
    MODEL = "model"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataAsset:
    id: str
    name: str
    type: DataAssetType
    source_system: str
    created_at: datetime
    last_updated: datetime
    quality_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    metadata: Dict
    
    def to_dict(self):
        return {
            **asdict(self),
            'type': self.type.value,
            'risk_level': self.risk_level.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class DataLineageTracker:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.assets: Dict[str, DataAsset] = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'validity': 0.90,
            'consistency': 0.85,
            'timeliness': 0.90
        }
    
    def add_asset(self, asset: DataAsset) -> None:
        """Add a data asset to the lineage graph"""
        self.assets[asset.id] = asset
        self.graph.add_node(asset.id, **asset.to_dict())
    
    def add_dependency(self, source_id: str, target_id: str, 
                      transformation: str = "", confidence: float = 1.0) -> None:
        """Add a dependency relationship between data assets"""
        if source_id not in self.assets or target_id not in self.assets:
            raise ValueError("Both source and target assets must exist")
        
        self.graph.add_edge(source_id, target_id, 
                           transformation=transformation,
                           confidence=confidence,
                           created_at=datetime.now().isoformat())
    
    def get_upstream_dependencies(self, asset_id: str, max_depth: int = 10) -> List[str]:
        """Get all upstream dependencies for an asset"""
        if asset_id not in self.graph:
            return []
        
        upstream = []
        current_level = [asset_id]
        
        for _ in range(max_depth):
            next_level = []
            for node in current_level:
                predecessors = list(self.graph.predecessors(node))
                upstream.extend(predecessors)
                next_level.extend(predecessors)
            
            if not next_level:
                break
            current_level = next_level
        
        return list(set(upstream))
    
    def get_downstream_impacts(self, asset_id: str, max_depth: int = 10) -> List[str]:
        """Get all downstream assets impacted by this asset"""
        if asset_id not in self.graph:
            return []
        
        downstream = []
        current_level = [asset_id]
        
        for _ in range(max_depth):
            next_level = []
            for node in current_level:
                successors = list(self.graph.successors(node))
                downstream.extend(successors)
                next_level.extend(successors)
            
            if not next_level:
                break
            current_level = next_level
        
        return list(set(downstream))
    
    def calculate_risk_propagation(self, asset_id: str) -> Dict[str, float]:
        """Calculate how risk propagates through the lineage"""
        if asset_id not in self.assets:
            return {}
        
        source_asset = self.assets[asset_id]
        base_risk = self._risk_level_to_score(source_asset.risk_level)
        
        downstream = self.get_downstream_impacts(asset_id)
        risk_scores = {asset_id: base_risk}
        
        for downstream_id in downstream:
            # Risk diminishes with distance but compounds with quality issues
            path_length = nx.shortest_path_length(self.graph, asset_id, downstream_id)
            downstream_asset = self.assets[downstream_id]
            
            # Risk calculation: base_risk * (1/path_length) * (1-quality_score)
            propagated_risk = base_risk * (1.0 / path_length) * (1.0 - downstream_asset.quality_score)
            risk_scores[downstream_id] = min(propagated_risk, 1.0)
        
        return risk_scores
    
    def identify_critical_paths(self) -> List[List[str]]:
        """Identify critical data paths with high risk or low quality"""
        critical_paths = []
        
        # Find paths where multiple high-risk assets are connected
        high_risk_assets = [
            asset_id for asset_id, asset in self.assets.items()
            if asset.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        for source in high_risk_assets:
            for target in high_risk_assets:
                if source != target:
                    try:
                        path = nx.shortest_path(self.graph, source, target)
                        if len(path) > 2:  # Not directly connected
                            critical_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return critical_paths
    
    def get_quality_impact_analysis(self, asset_id: str) -> Dict:
        """Analyze quality impact of an asset on downstream systems"""
        if asset_id not in self.assets:
            return {}
        
        asset = self.assets[asset_id]
        downstream = self.get_downstream_impacts(asset_id)
        
        analysis = {
            'source_asset': asset.to_dict(),
            'affected_assets': len(downstream),
            'quality_degradation_risk': 1.0 - asset.quality_score,
            'impacted_models': [],
            'impacted_outputs': []
        }
        
        for downstream_id in downstream:
            downstream_asset = self.assets[downstream_id]
            if downstream_asset.type == DataAssetType.MODEL:
                analysis['impacted_models'].append(downstream_asset.to_dict())
            elif downstream_asset.type == DataAssetType.OUTPUT:
                analysis['impacted_outputs'].append(downstream_asset.to_dict())
        
        return analysis
    
    def generate_lineage_report(self) -> Dict:
        """Generate comprehensive lineage report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_assets': len(self.assets),
            'total_dependencies': self.graph.number_of_edges(),
            'asset_types': {},
            'risk_distribution': {},
            'quality_summary': {
                'average_quality': 0.0,
                'below_threshold_assets': []
            },
            'critical_paths': self.identify_critical_paths(),
            'isolated_assets': list(nx.isolates(self.graph))
        }
        
        # Asset type distribution
        for asset in self.assets.values():
            asset_type = asset.type.value
            report['asset_types'][asset_type] = report['asset_types'].get(asset_type, 0) + 1
        
        # Risk distribution
        for asset in self.assets.values():
            risk_level = asset.risk_level.value
            report['risk_distribution'][risk_level] = report['risk_distribution'].get(risk_level, 0) + 1
        
        # Quality summary
        quality_scores = [asset.quality_score for asset in self.assets.values()]
        report['quality_summary']['average_quality'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        report['quality_summary']['below_threshold_assets'] = [
            asset.to_dict() for asset in self.assets.values()
            if asset.quality_score < 0.8  # Configurable threshold
        ]
        
        return report
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numerical score"""
        risk_mapping = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 1.0
        }
        return risk_mapping.get(risk_level, 0.5)
    
    def export_to_json(self, filepath: str) -> None:
        """Export lineage graph to JSON"""
        export_data = {
            'assets': {asset_id: asset.to_dict() for asset_id, asset in self.assets.items()},
            'dependencies': [
                {
                    'source': source,
                    'target': target,
                    'attributes': data
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_from_json(self, filepath: str) -> None:
        """Import lineage graph from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Import assets
        for asset_data in data['assets'].values():
            asset_data['type'] = DataAssetType(asset_data['type'])
            asset_data['risk_level'] = RiskLevel(asset_data['risk_level'])
            asset_data['created_at'] = datetime.fromisoformat(asset_data['created_at'])
            asset_data['last_updated'] = datetime.fromisoformat(asset_data['last_updated'])
            
            asset = DataAsset(**asset_data)
            self.add_asset(asset)
        
        # Import dependencies
        for dep in data['dependencies']:
            self.graph.add_edge(dep['source'], dep['target'], **dep['attributes'])

# Example usage and demonstration
if __name__ == "__main__":
    tracker = DataLineageTracker()
    
    # Create sample assets
    raw_data = DataAsset(
        id="customer_raw_data",
        name="Customer Raw Data",
        type=DataAssetType.SOURCE,
        source_system="CRM",
        created_at=datetime.now() - timedelta(days=30),
        last_updated=datetime.now() - timedelta(days=1),
        quality_score=0.85,
        risk_level=RiskLevel.MEDIUM,
        metadata={"format": "CSV", "size_gb": 2.5}
    )
    
    cleaned_data = DataAsset(
        id="customer_cleaned_data",
        name="Customer Cleaned Data",
        type=DataAssetType.TRANSFORMATION,
        source_system="Data Pipeline",
        created_at=datetime.now() - timedelta(days=29),
        last_updated=datetime.now() - timedelta(hours=6),
        quality_score=0.92,
        risk_level=RiskLevel.LOW,
        metadata={"transformation": "data_cleaning_v2"}
    )
    
    ml_model = DataAsset(
        id="churn_prediction_model",
        name="Customer Churn Prediction Model",
        type=DataAssetType.MODEL,
        source_system="ML Platform",
        created_at=datetime.now() - timedelta(days=7),
        last_updated=datetime.now() - timedelta(hours=12),
        quality_score=0.88,
        risk_level=RiskLevel.HIGH,
        metadata={"model_type": "RandomForest", "accuracy": 0.94}
    )
    
    # Add assets to tracker
    tracker.add_asset(raw_data)
    tracker.add_asset(cleaned_data)
    tracker.add_asset(ml_model)
    
    # Add dependencies
    tracker.add_dependency("customer_raw_data", "customer_cleaned_data", "data_cleaning", 0.95)
    tracker.add_dependency("customer_cleaned_data", "churn_prediction_model", "model_training", 0.90)
    
    # Analysis
    print("=== Data Lineage Analysis ===")
    print(f"Upstream dependencies of ML model: {tracker.get_upstream_dependencies('churn_prediction_model')}")
    print(f"Risk propagation from raw data: {tracker.calculate_risk_propagation('customer_raw_data')}")
    
    # Generate report
    report = tracker.generate_lineage_report()
    print(f"\nLineage Report Summary:")
    print(f"- Total assets: {report['total_assets']}")
    print(f"- Total dependencies: {report['total_dependencies']}")
    print(f"- Average quality score: {report['quality_summary']['average_quality']:.2f}")
    print(f"- Critical paths found: {len(report['critical_paths'])}")