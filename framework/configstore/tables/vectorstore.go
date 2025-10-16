package tables

import "time"

// TableVectorStoreConfig represents Cache plugin configuration in the database
type TableVectorStoreConfig struct {
	ID              uint      `gorm:"primaryKey;autoIncrement" json:"id"`
	Enabled         bool      `json:"enabled"`                               // Enable vector store
	Type            string    `gorm:"type:varchar(50);not null" json:"type"` // "weaviate, elasticsearch, pinecone, etc."
	TTLSeconds      int       `gorm:"default:300" json:"ttl_seconds"`        // TTL in seconds (default: 5 minutes)
	CacheByModel    bool      `gorm:"" json:"cache_by_model"`                // Include model in cache key
	CacheByProvider bool      `gorm:"" json:"cache_by_provider"`             // Include provider in cache key
	Config          *string   `gorm:"type:text" json:"config"`               // JSON serialized schemas.RedisVectorStoreConfig
	CreatedAt       time.Time `gorm:"index;not null" json:"created_at"`
	UpdatedAt       time.Time `gorm:"index;not null" json:"updated_at"`
}

// TableName sets the table name for each model
func (TableVectorStoreConfig) TableName() string { return "config_vector_store" }
